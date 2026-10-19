"""Reproducible local-model smoke checks; reports omit keys, endpoints and pixels."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['embedding', 'vision'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='outputs/acceptance/core_upgrade/first_pass')
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    images = sorted((ROOT / 'data/ncpa_test/京剧《天下归心》').rglob('*.jpg'))
    if not images:
        raise RuntimeError('需要真实剧照样本')
    report = {'component': args.component, 'scope': 'smoke_only', 'sample_count': 1}
    start = time.perf_counter()
    try:
        if args.component == 'embedding':
            import numpy as np
            import torch
            import transformers
            import sentence_transformers
            from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder
            model = EmbeddingGemmaEmbedder(device=args.device)
            sample = [str(p) for p in images[:3]]
            vectors = model.extract_features_batch(sample)
            sequential = np.stack([model.extract_features(p) for p in sample])
            texts = model.encode_text_batch(['戏剧舞台上身穿戏服的演员', '海边的沙滩和汽车'])
            from core_modules.organize.multimodal_evidence import MultimodalRetriever
            knowledge = {'scenes': [{'scene_id': 'act-01', 'act': 1, 'label': '戏剧', 'text': '舞台演员'},
                                    {'scene_id': 'act-02', 'act': 2, 'label': '海滩', 'text': '沙滩汽车'}]}
            retriever = MultimodalRetriever(model, cache_dir=str(target / 'embedding_cache'))
            retriever.prepare(sample, knowledge)
            retriever.save()
            cached = MultimodalRetriever(model, cache_dir=str(target / 'embedding_cache'))
            cached.prepare(sample, knowledge)
            report.update(model_signature=model.model_signature, image_shape=list(vectors.shape),
                          text_shape=list(texts.shape), finite=bool(np.isfinite(vectors).all() and np.isfinite(texts).all()),
                          cosine_scores=(vectors @ texts.T).tolist(), dtype=str(next(model._model.parameters()).dtype),
                          transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__,
                          sample_count=len(sample), min_batch_single_cosine=float((vectors * sequential).sum(axis=1).min()),
                          cached_forward_calls=cached.calls)
            report['status'] = 'passed' if report['finite'] and report['min_batch_single_cosine'] >= .999 and cached.calls == {'image': 0, 'text': 0} else 'failed'
        else:
            from PIL import Image, ImageOps
            from core_modules.organize.llm_client import vision_client_from_env
            from core_modules.organize.scene_reasoner import SceneReasoner, validate_choice
            client = vision_client_from_env()
            if client is None:
                raise RuntimeError('大模型端点配置缺失')
            client.timeout = 45
            client.max_retries = 0
            responses = []
            original_opener = client._opener
            def diagnostic_opener(request, timeout):
                with original_opener(request, timeout=timeout) as response:
                    raw = response.read()
                data = json.loads(raw)
                choice = data.get('choices', [{}])[0]
                message = choice.get('message') or {}
                responses.append({'finish_reason': choice.get('finish_reason'),
                                  'content_type': type(message.get('content')).__name__,
                                  'content_characters': len(message.get('content') or ''),
                                  'usage': data.get('usage')})
                return io.BytesIO(raw)
            client._opener = diagnostic_opener
            report['responses'] = responses
            knowledge = json.loads((ROOT / 'outputs/play_knowledge/txgx201410/entities.json').read_text())
            # Exercise both requests with neutral metadata, so the filename
            # shortcut cannot masquerade as successful visual reasoning.
            with tempfile.TemporaryDirectory(prefix='theatre-vision-smoke-') as temporary:
                neutral = Path(temporary) / 'photo.png'
                with Image.open(images[0]) as original:
                    preview = ImageOps.exif_transpose(original).convert('RGB')
                    preview.thumbnail((1280, 1280))
                    preview.info.clear()
                    preview.save(neutral)
                row = SceneReasoner(knowledge, client=client, vision_client=client,
                                    use_rule_shortcut=False).reason(str(neutral))
            evidence = next((e for e in row['evidence'] if e['type'] == 'observation'), None)
            trace = row['model_trace']
            choice = trace.get('choice')
            valid_choice, _ = validate_choice(choice, [s['scene_id'] for s in knowledge['scenes']],
                                              [e['id'] for e in row['evidence']])
            report.update(served_model=client.model, model_identity=client.model_identity,
                          model_identity_source=client.model_identity_source,
                          valid_observation=evidence is not None,
                          valid_closed_set_choice=valid_choice, scene_id=row['scene_id'],
                          review_status=row['review']['status'], neutral_input=True,
                          scope='one_image_observation_and_closed_set_smoke_only',
                          status='passed' if evidence and valid_choice and row['review']['status'] == 'pending' else 'failed')
            if evidence:
                report['observation'] = json.loads(evidence['quote'])
            report['trace'] = {stage: {key: values[key] for key in (
                'model', 'model_identity', 'model_identity_source', 'prompt_version',
                'prompt_hash', 'response_hash', 'elapsed_ms', 'error', 'rejected') if key in values}
                for stage, values in [('vision', trace.get('vision', {})), ('language', trace)]}
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        report['error_locations'] = [f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}'
                                     for frame in traceback.extract_tb(exc.__traceback__)]
        # Import/loading messages are safe; network exceptions may contain configured endpoints.
        if args.component == 'embedding':
            report['error'] = str(exc)[:1000]
    report['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    path = target / f'{args.component}_smoke.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
