"""List/prune old image collections owned by indexes under an explicit root.

Current and previous published generations are retained. A minimum age and the
same per-index writer lock protect unfinished builds. Default is a dry run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.qdrant_images import index_lock, load_manifest, MANIFEST
from core_modules.vector_store import QdrantStore


def prune(root, minimum_age_hours=24, apply=False, store=None):
    if minimum_age_hours < 1:
        raise ValueError('minimum_age_must_be_at_least_one_hour')
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('index_root_not_found')
    store = store or QdrantStore()
    now = datetime.now(timezone.utc)
    rows = []
    for row in store.request('GET', '/collections')['collections']:
        name = row['name']
        metadata = store.info(name)['config'].get('metadata') or {}
        if not metadata.get('owner_index') or not metadata.get('created_at'):
            continue
        owner = Path(metadata['owner_index']).resolve()
        if not owner.is_relative_to(root) or not (owner / MANIFEST).is_file():
            continue
        age = (now - datetime.fromisoformat(metadata['created_at'])).total_seconds() / 3600
        if age < minimum_age_hours:
            continue
        with index_lock(owner):
            current = load_manifest(owner)
            if name in (current['collection'], current.get('previous_collection')):
                continue
            if apply:
                store.delete_collection(name)
            rows.append({'collection': name, 'index_path': str(owner), 'age_hours': age, 'deleted': apply})
    return {'dry_run': not apply, 'candidates': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--minimum-age-hours', type=float, default=24)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prune(args.root, args.minimum_age_hours, args.apply), ensure_ascii=False, indent=2))
