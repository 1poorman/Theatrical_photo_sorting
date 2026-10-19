"""Exploratory image-only scene retrieval on stripped, neutrally named previews.

    Filename-derived labels are provisional. This is not a held-out final test.
    Run each backend in its compatible environment; both use the same manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_scenes(knowledge_dir):
    # scene_candidates.jsonl contains one retrieval result per image, not scenes.
    scenes = json.loads((Path(knowledge_dir) / 'entities.json').read_text())['scenes']
    if not scenes or any(not s.get('scene_id') or not s.get('label') for s in scenes):
        raise ValueError('知识库没有有效幕次；请检查 entities.json')
    if len({s['scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('知识库幕次 ID 重复')
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['siglip2_base', 'embeddinggemma2'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='outputs/acceptance/core_upgrade/first_pass')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats 必须大于零')
    output = Path(args.output)
    previews = output / 'neutral_images'
    previews.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/ncpa_test/京剧《天下归心》/【整理完成】京剧《天下归心》彩排'
    from core_modules.organize.scene_reasoner import _parse_local
    scenes = load_scenes(ROOT / 'outputs/play_knowledge/txgx201410')
    manifest = []
    excluded = []
    for original in sorted(source.glob('*.jpg')):
        scene = _parse_local(str(original)).get('scene') or ''
        matches = [s for s in scenes if s.get('title') and s['title'] in scene]
        if len(matches) != 1:
            excluded.append({'source_hash': hashlib.sha256(original.read_bytes()).hexdigest(),
                             'reason': 'missing_or_ambiguous_provisional_label'})
            continue
        with Image.open(original) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1280, 1280))
            # PIL convert/transpose keep info (including ICC profiles and EXIF).
            image.info.clear()
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            name = f'{digest[:24]}.png'
            image.save(previews / name)
        if any(row['content_hash'] == digest for row in manifest):
            continue
        manifest.append({'image_id': digest[:24], 'content_hash': digest,
                         'image': name, 'expected': matches[0]['scene_id'],
                         'ground_truth_source': 'provisional_filename_label'})
    (output / 'exploratory_manifest.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in manifest))
    if not manifest:
        raise ValueError('没有可评测样本，停止加载模型；请核对图片目录、幕次表和弱标签')
    # Freeze the actual text pool alongside the image manifest for both backends.
    write_json(output / 'exploratory_scenes.json', scenes)
    texts = [s.get('label', '') + ' ' + s.get('text', '') for s in scenes]
    if args.backend == 'embeddinggemma2':
        from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder
        model = EmbeddingGemmaEmbedder(device=args.device)
        model._load()
    else:
        from core_modules.image_search import ImageEmbedder
        model = ImageEmbedder(model_name='siglip2_base', device=args.device)
    import torch
    import transformers
    cuda = str(args.device).startswith('cuda')
    def synchronize():
        if cuda:
            torch.cuda.synchronize(args.device)

    paths = [str((previews / row['image']).resolve()) for row in manifest]
    def encode(image_paths, scene_texts):
        if args.backend == 'embeddinggemma2':
            return model.extract_features_batch(image_paths), model.encode_text_batch(scene_texts)
        images, valid = model.extract_features_batch(image_paths, sub_batch_size=8)
        if valid != image_paths:
            raise ValueError('部分图片编码失败，不能缩小评测分母')
        return images, np.stack([model.encode_text(t) for t in scene_texts])

    encode(paths[:1], texts[:1])
    synchronize()
    if cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    times = []
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        images, documents = encode(paths, texts)
        synchronize()
        times.append(time.perf_counter() - start)
    def normalize(vectors, expected):
        vectors = np.asarray(vectors, dtype='float32')
        if vectors.ndim != 2 or len(vectors) != expected or not np.isfinite(vectors).all():
            raise ValueError('向量数量、维度或有限性检查失败')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError('向量中存在零向量')
        return vectors / norms
    images = normalize(images, len(paths))
    documents = normalize(documents, len(texts))
    scores = images @ documents.T
    ranking = np.argsort(-scores, axis=1)
    ids = [s['scene_id'] for s in scenes]
    rows = []
    for index, row in enumerate(manifest):
        ordered = [ids[i] for i in ranking[index]]
        rows.append({'image_id': row['image_id'], 'expected': row['expected'],
                     'ranked_scenes': ordered, 'scores': [float(scores[index, i]) for i in ranking[index]]})
    report = {'backend': args.backend, 'status': 'insufficient_data',
              'scope': 'single_play_exploratory_image_only', 'sample_count': len(rows),
              'scene_count': len(scenes), 'neutral_names': True, 'metadata_stripped': True,
              'neutral_preview_version': 2,
              'ground_truth': 'provisional_filename_labels_not_independently_reviewed',
              'recall_at_1': sum(r['expected'] == r['ranked_scenes'][0] for r in rows) / len(rows),
              'recall_at_3': sum(r['expected'] in r['ranked_scenes'][:3] for r in rows) / len(rows),
              'hits_at_1': sum(r['expected'] == r['ranked_scenes'][0] for r in rows),
              'hits_at_3': sum(r['expected'] in r['ranked_scenes'][:3] for r in rows),
              'repeats': args.repeats, 'cache_state': 'warmed_process_no_feature_cache',
              'elapsed_seconds': [round(t, 3) for t in times],
              'median_seconds': round(statistics.median(times), 3),
              'timing_scope': 'neutral_preview_image_and_scene_text_encoding_only',
              'model_signature': model.model_signature,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'transformers': transformers.__version__, 'device': args.device,
                              'gpu': torch.cuda.get_device_name(args.device) if cuda else None,
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if cuda else None},
              'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              'manifest_hash': hashlib.sha256((output / 'exploratory_manifest.jsonl').read_bytes()).hexdigest(),
              'text_hash': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
              'excluded': excluded, 'rows': rows}
    write_json(output / f'{args.backend}_retrieval.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
