"""Multi-play image-only retrieval with user-confirmed image-level labels.

All annotated scene labels are in the pool, including curtain calls. Query
names/EXIF are stripped; actor lists and image captions never enter documents.
This is a development comparison: the historical split is not a locked test.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.organize.identity_registry import IdentityRegistry
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from tools.sourced_scene_candidates import (apply_candidate_bundle, candidate_files_unchanged,
                                           validate_sourced_manifest)


def normalized_label(value):
    import re
    return re.sub(r'[\s：:·]', '', value)


def prepare_inputs(labels_path, registry_path, source_root, output):
    from core_modules.organize.play_knowledge import parse_plot_text
    labels = json.loads(Path(labels_path).read_text())
    registry = IdentityRegistry.load(registry_path)
    source_root = Path(source_root).resolve()
    previews = output / 'neutral_images'
    previews.mkdir(parents=True, exist_ok=True)
    by_play, excluded, scenes = defaultdict(list), [], []
    seen, expected_by_image = set(), {}
    for row in labels:
        if not row['scene_id'] or row['status'] != 'ok':
            excluded.append({'file_id': row['file_id'], 'reason': 'no_scene_label_or_decode_failed'})
            continue
        if row['image_id'] in expected_by_image and expected_by_image[row['image_id']] != (row['play_id'],row['scene_id']):
            raise ValueError('conflicting_duplicate_label')
        expected_by_image[row['image_id']] = (row['play_id'],row['scene_id'])
        if row['image_id'] in seen:
            excluded.append({'file_id':row['file_id'],'reason':'duplicate_decoded_image'})
            continue
        seen.add(row['image_id'])
        path = source_root / row['source']
        if fingerprint(path) != row['content_hash']:
            raise ValueError('annotated_photo_content_changed')
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1280,1280))
            image.info.clear()
            neutral = previews / (row['image_id'][:24]+'.png')
            image.save(neutral)
        by_play[row['play_id']].append({**row, 'neutral_path':str(neutral.resolve()),
                                       'neutral_sha256':fingerprint(neutral)})
    for play_id, rows in by_play.items():
        original_directory = source_root / Path(rows[0]['source']).parts[0]
        plot_files = [p for p in sorted(original_directory.glob('剧目信息*')) if p.is_file()]
        parsed_scenes = []
        for path in plot_files:
            parsed_scenes.extend(parse_plot_text(path.read_text(encoding='utf-8'),str(path.relative_to(source_root)))[1])
        unique = {r['scene_id']:r['scene_label'] for r in rows}
        for scene_id,label in sorted(unique.items()):
            matches = [scene for scene in parsed_scenes
                       if normalized_label(scene['label'])==normalized_label(label)]
            text = matches[0].get('text','') if len(matches)==1 else ''
            scenes.append({'play_id':play_id,'scene_id':scene_id,'label':label,
                           'text':registry.plays[play_id]+' '+label+' '+text,
                           'text_source':'local_plot_and_scene_label' if text else 'scene_label_only',
                           'source_ref':matches[0]['source_ref'] if len(matches)==1 else 'user_confirmed_scene_vocabulary'})
    if not by_play:
        raise ValueError('no_annotated_scene_queries')
    atomic_json(output/'input_manifest.json',{'labels_sha256':fingerprint(labels_path),
                'registry_sha256':fingerprint(registry_path),'rows':[r for rows in by_play.values() for r in rows],
                'scenes':scenes,'excluded':excluded,'test_locked':False})
    return by_play,scenes,excluded


def normalize(values, count):
    values=np.asarray(values,dtype='float32')
    if values.ndim!=2 or len(values)!=count or not np.isfinite(values).all():
        raise ValueError('invalid_embedding_shape_or_value')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if (norms<=0).any():
        raise ValueError('zero_embedding')
    return values/norms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT/'data/ncpa_test')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('siglip2_base','embeddinggemma2'),required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--prepared-inputs',type=Path,help='Reuse the exact saved neutral input manifest for a paired backend run')
    parser.add_argument('--candidate-texts',type=Path,
                        help='Hash-bound HTML excerpts mapped to the saved vocabulary; requires --prepared-inputs')
    args=parser.parse_args()
    if args.candidate_texts and not args.prepared_inputs:
        parser.error('--candidate-texts requires --prepared-inputs')
    output=args.output.resolve(); source=args.source_root.resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('output_must_be_outside_originals')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_must_be_empty')
    output.mkdir(parents=True,exist_ok=True)
    versions={name:fingerprint(ROOT/name) for name in (
        'tools/evaluate_verified_scene_retrieval.py','tools/sourced_scene_candidates.py',
        'core_modules/organize/identity_registry.py',
        'core_modules/organize/play_knowledge.py','core_modules/organize/filename_parser.py',
        'core_modules/image_search.py','core_modules/organize/embedding_gemma.py',
        'core_modules/organize/feature_cache.py','config/base.py')}
    if args.prepared_inputs:
        prepared_hash=fingerprint(args.prepared_inputs)
        manifest=json.loads(args.prepared_inputs.read_text())
        if manifest['labels_sha256']!=fingerprint(args.labels) or manifest['registry_sha256']!=fingerprint(args.registry):
            raise ValueError('prepared_annotations_changed')
        registry=IdentityRegistry.load(args.registry)
        validate_sourced_manifest(manifest,registry)
        if args.candidate_texts:
            manifest=apply_candidate_bundle(manifest,args.prepared_inputs,args.candidate_texts,registry)
        by_play=defaultdict(list)
        for row in manifest['rows']:
            if fingerprint(source/row['source'])!=row['content_hash'] or fingerprint(row['neutral_path'])!=row['neutral_sha256']:
                raise ValueError('prepared_image_changed')
            by_play[row['play_id']].append(row)
        scenes,excluded=manifest['scenes'],manifest['excluded']
        atomic_json(output/'input_manifest.json',manifest)
    else:
        by_play,scenes,excluded=prepare_inputs(args.labels,args.registry,args.source_root,output)
        manifest=json.loads((output/'input_manifest.json').read_text())
    labels_hash,registry_hash=fingerprint(args.labels),fingerprint(args.registry)
    print(json.dumps({'query_count':sum(map(len,by_play.values())),'plays':len(by_play),'scene_pool_size':len(scenes),
                     'text_sources':dict(Counter(s['text_source'] for s in scenes))}),flush=True)
    if args.backend=='siglip2_base':
        from core_modules.image_search import ImageEmbedder
        model=ImageEmbedder('siglip2_base',device=args.device)
        if model.processor is None or getattr(getattr(model.model,'config',None),'model_type',None) not in ('siglip','siglip2'):
            raise ValueError('siglip_weights_not_loaded')
    else:
        from core_modules.organize.embedding_gemma import EmbeddingGemmaEmbedder
        model=EmbeddingGemmaEmbedder(device=args.device)
    started=time.perf_counter()
    documents=normalize(model.encode_text_batch([s['text'] for s in scenes]),len(scenes))
    results=[]
    for play_id,rows in by_play.items():
        pool=[i for i,s in enumerate(scenes) if s['play_id']==play_id]
        paths=[r['neutral_path'] for r in rows]
        if args.backend=='siglip2_base':
            vectors,valid=model.extract_features_batch(paths,sub_batch_size=8)
            if valid!=paths:raise ValueError('failed_image_not_removed_from_denominator')
        else:
            vectors=model.extract_features_batch(paths)
        scores=normalize(vectors,len(paths)) @ documents[pool].T
        for row,values in zip(rows,scores):
            ranking=np.argsort(-values,kind='stable')
            ranked=[scenes[pool[i]]['scene_id'] for i in ranking]
            results.append({'file_id':row['file_id'],'play_id':play_id,'expected':row['scene_id'],
                            'ranked_scenes':ranked,'scores':[float(values[i]) for i in ranking]})
        print(json.dumps({'play_id':play_id,'queries':len(rows),'pool':len(pool)},ensure_ascii=False),flush=True)
    def metrics(rows):
        counts=Counter(r['expected'] for r in rows)
        return {'count':len(rows),'recall_at_1':sum(r['expected']==r['ranked_scenes'][0] for r in rows)/len(rows),
                'recall_at_3':sum(r['expected'] in r['ranked_scenes'][:3] for r in rows)/len(rows),
                'scene_macro_recall_at_1':sum(sum(r['expected']==s and r['ranked_scenes'][0]==s for r in rows)/n
                                            for s,n in counts.items())/len(counts)}
    checks={'originals_unchanged':all(fingerprint(source/r['source'])==r['content_hash'] for rows in by_play.values() for r in rows),
            'neutral_previews_unchanged':all(fingerprint(r['neutral_path'])==r['neutral_sha256'] for rows in by_play.values() for r in rows),
            'code_unchanged':all(fingerprint(ROOT/n)==h for n,h in versions.items()),
            'all_labeled_queries_retained':len(results)==sum(map(len,by_play.values())),
            'annotations_unchanged':fingerprint(args.labels)==labels_hash and fingerprint(args.registry)==registry_hash,
            'candidate_sources_unchanged':candidate_files_unchanged(manifest),
            'prepared_manifest_unchanged':not args.prepared_inputs or fingerprint(args.prepared_inputs)==prepared_hash}
    report={'backend':args.backend,'status':'development_comparison_final_test_not_locked',
            'metrics':metrics(results),'per_play':{p:metrics([r for r in results if r['play_id']==p]) for p in by_play},
            'source_sha256':versions,'manifest_sha256':fingerprint(output/'input_manifest.json'),
            'model_signature':model.model_signature,'preprocessing_signature':model.preprocessing_signature,
            'candidate_texts':manifest.get('candidate_texts'),
            'encoding_elapsed_seconds':time.perf_counter()-started,
            'checks':checks,'rows':results,'excluded':excluded,
            'limitations':['User-confirmed scene labels; not a locked held-out final test.',
                'Whole-play pool retains confirmed label vocabulary and any explicitly sourced extra scenes; query captions excluded.',
                'Local paragraphs or hash-bound source excerpts; label-only candidates explicitly recorded.',
                'Three-class pools make Recall@3 trivial; inspect Recall@1 and per-play results.',
                'Shared GPU, one encoding pass; no U2 performance claim.']}
    atomic_json(output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('backend','metrics','per_play','checks')},ensure_ascii=False,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':
    main()
