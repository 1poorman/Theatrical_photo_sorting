"""Real container HTTP acceptance for online vectors, migration and restart checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode
import uuid

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core_modules.qdrant_images import load_manifest, fingerprint, MANIFEST
from core_modules.vector_store import QdrantStore, match_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8199')
    parser.add_argument('--after-restart', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / 'state.json'
    checks = []
    state = json.loads(state_file.read_text()) if args.after_restart else {}
    store = QdrantStore()
    client = httpx.Client(base_url=args.url, timeout=300, trust_env=False)

    def api(path, form=None, payload=None, file=None, expected=200):
        if file:
            with open(file, 'rb') as stream:
                field = 'image' if path == '/api/face/recognize' else 'query_image'
                response = client.post(path, data=form or {}, files={field: ('acceptance.jpg', stream, 'image/jpeg')})
        elif payload is not None:
            response = client.post(path, json=payload)
        elif form is not None:
            response = client.post(path, data=form)
        else:
            response = client.get(path)
        assert response.status_code == expected, (path, response.status_code, response.text[:1000])
        body = response.json()
        return body.get('data', body)

    def check(name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            row = {'name': name, 'passed': True, 'detail': detail}
        except Exception as exc:
            row = {'name': name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        row['seconds'] = round(time.perf_counter()-start, 3)
        checks.append(row)
        target = out / ('restart_report.json' if args.after_restart else 'report.json')
        target.write_text(json.dumps({'checks': checks, 'passed': sum(c['passed'] for c in checks),
            'total': len(checks), 'all_passed': all(c['passed'] for c in checks)}, ensure_ascii=False, indent=2))
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def fixture():
        source = ROOT / 'data/ncpa_test/京剧《天下归心》/【整理完成】京剧《天下归心》彩排'
        photos = out / 'photos'
        photos.mkdir()
        for p in sorted(source.glob('*.jpg'))[:3]:
            with Image.open(p) as image:
                image = image.convert('RGB')
                image.thumbnail((768, 768))
                image.save(photos / p.name, quality=95)
        state.update(photos=str(photos), query=str(next(photos.glob('*.jpg'))), source=str(source))
        return {'photos': 3}

    def indexes():
        old = api('/api/embedding/build_index', form={'image_folder': state['photos'],
            'index_save_path': str(out / 'faiss'), 'model_name': 'siglip2_base', 'backend': 'faiss'})
        new = api('/api/embedding/build_index', form={'image_folder': state['photos'],
            'index_save_path': str(out / 'qdrant'), 'model_name': 'siglip2_base',
            'metadata_json': json.dumps({'project_id': 'acceptance'})})
        assert old['backend'] == 'faiss' and new['backend'] == 'qdrant'
        assert old['image_count'] == new['image_count'] == 3
        state.update(faiss=old['index_path'], qdrant=new['index_path'])
        assert not Path(new['index_path'], 'embeddings.npy').exists()
        assert not Path(new['index_path'], 'vector_index.faiss').exists()
        status = api('/api/embedding/index?' + urlencode({'index_path': new['index_path']}))
        assert status['image_count'] == 3
        return {'legacy_and_qdrant_built': True, 'manifest_bytes': Path(new['index_path'], MANIFEST).stat().st_size}

    def search():
        old = api('/api/embedding/search', form={'index_path': state['faiss'], 'top_k': 3}, file=state['query'])['results']
        new = api('/api/embedding/search', form={'index_path': state['qdrant'], 'top_k': 3, 'exact': 'true'}, file=state['query'])['results']
        assert [r['image_path'] for r in old] == [r['image_path'] for r in new]
        error = max(abs(a['similarity']-b['similarity']) for a, b in zip(old,new))
        assert error < 1e-5, error
        state['expected_paths'] = [r['image_path'] for r in new]
        return {'top3_identical': True, 'max_cosine_error': error}

    def incremental():
        extra = out / 'extra.jpg'
        shutil.copy2(state['query'], extra)
        payload = {'index_path': state['qdrant'], 'image_paths': [str(extra)], 'metadata': {'project_id': 'incremental'}}
        for _ in range(2):
            data = api('/api/embedding/upsert', payload=payload)
            assert data['image_count'] == 4 and data['upserted'] == 1
        # An existing path's vector changes, without adding another point or losing metadata.
        shutil.copy2(sorted(Path(state['photos']).glob('*.jpg'))[-1], extra)
        api('/api/embedding/upsert', payload={'index_path': state['qdrant'], 'image_paths': [str(extra)]})
        result = api('/api/embedding/search', form={'index_path': state['qdrant'], 'top_k': 5,
            'filters_json': json.dumps({'project_id': 'incremental'})}, file=extra)['results']
        assert len(result) == 1 and result[0]['image_path'] == str(extra) and result[0]['similarity'] > .999
        deleted = api('/api/embedding/delete', payload={'index_path': state['qdrant'], 'image_paths': [str(extra)]})
        assert deleted == {'deleted': 1, 'image_count': 3}
        assert api('/api/embedding/search', form={'index_path': state['qdrant'],
            'filters_json': json.dumps({'project_id': 'incremental'})}, file=extra)['results'] == []
        return {'idempotent_upsert': True, 'changed_vector_visible': True, 'metadata_preserved': True,
                'filtered_search_and_delete': True}

    def migrate():
        from tools.migrate_vector_store import migrate_faiss
        manifest = load_manifest(state['qdrant'])
        model = SimpleNamespace(model_name=manifest['model_name'], feature_dim=manifest['dimension'],
            model_signature=manifest['model_signature'], preprocessing_signature=manifest['preprocessing_signature'])
        destination = out / 'migrated'
        report = migrate_faiss(state['faiss'], destination, model)
        state['migrated'] = str(destination)
        hits = api('/api/embedding/search', form={'index_path': str(destination), 'top_k': 3, 'exact': 'true'}, file=state['query'])['results']
        assert [h['image_path'] for h in hits] == state['expected_paths']
        return report

    def failure():
        api('/api/embedding/search', form={'index_path': state['qdrant'], 'filters_json': '{bad'}, file=state['query'], expected=400)
        api('/api/embedding/upsert', payload={'index_path': state['qdrant'], 'image_paths': []}, expected=400)
        directory = out / 'broken_reference'
        directory.mkdir()
        manifest = load_manifest(state['qdrant'])
        manifest['collection'] = 'missing_' + uuid.uuid4().hex
        (directory / MANIFEST).write_text(json.dumps(manifest))
        api('/api/embedding/search', form={'index_path': str(directory)}, file=state['query'], expected=503)
        assert api('/api/embedding/index?' + urlencode({'index_path': state['qdrant']}))['image_count'] == 3
        return {'invalid_requests_rejected': True, 'missing_collection_503_no_stale_fallback': True}

    def faces():
        from config.base import ARCFACE_MODEL_PATH
        from core_modules.organize.feature_cache import ContentHasher
        from core_modules.face_recognition.qdrant_database import collection_name
        person = 'qdrant_acceptance_' + uuid.uuid4().hex
        path = out / 'face_anchors' / person
        path.mkdir(parents=True)
        source = sorted(Path(state['source']).glob('*.jpg'))[2]
        with Image.open(source) as image:
            image = image.convert('RGB')
            image.thumbnail((1600,1600))
            image.save(path / 'anchor.jpg', quality=95)
        collection = collection_name(ContentHasher().fingerprint(ARCFACE_MODEL_PATH)[0])
        state.update(face_collection=collection, person=person, anchor=str(path / 'anchor.jpg'))
        for _ in range(2):
            api('/api/face/build_database', form={'face_db_folder': str(path.parent)})
        assert store.count(collection, match_filter({'person_id': person})) == 1
        result = api('/api/face/recognize', file=path / 'anchor.jpg')['recognized_faces']
        assert any(face['identified_as'] == person for face in result), result
        return {'face_backend': 'qdrant', 'duplicate_build_single_anchor': True, 'recognized_faces': len(result)}

    def prepare_restart():
        state['manifest_hash'] = fingerprint(Path(state['qdrant']) / MANIFEST)
        state['face_count'] = store.count(state['face_collection'], match_filter({'person_id': state['person']}))
        return {'references_saved': True, 'face_count': state['face_count']}

    def restart():
        assert fingerprint(Path(state['qdrant']) / MANIFEST) == state['manifest_hash']
        search()
        assert store.count(state['face_collection'], match_filter({'person_id': state['person']})) == state['face_count']
        result = api('/api/face/recognize', file=state['anchor'])['recognized_faces']
        assert any(face['identified_as'] == state['person'] for face in result)
        return {'image_and_face_queries_survive_restart': True, 'manifest_unchanged': True}

    def cleanup():
        store.delete_points(state['face_collection'], filters=match_filter({'person_id': state['person']}))
        assert store.count(state['face_collection'], match_filter({'person_id': state['person']})) == 0
        return {'temporary_face_records_removed': True, 'image_acceptance_collections_retained': True}

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            api('/api/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('HTTP service not ready')
    steps = [('restart_queries', restart), ('cleanup', cleanup)] if args.after_restart else [
        ('fixture', fixture), ('build_old_and_new', indexes), ('search_parity', search),
        ('incremental_update_filter_delete', incremental), ('legacy_migration', migrate),
        ('failure_handling', failure), ('real_face_database', faces), ('restart_snapshot', prepare_restart)]
    try:
        for name, fn in steps:
            check(name, fn)
    finally:
        client.close()
    raise SystemExit(0 if all(c['passed'] for c in checks) else 1)


if __name__ == '__main__':
    main()
