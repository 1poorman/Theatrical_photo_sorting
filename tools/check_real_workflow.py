"""Run reason -> isolated review -> export -> organize with actual local models.

This exercises workflow contracts, not identity/scene accuracy or U2 speed.
Scripted review decisions only touch the new acceptance directory.
"""
import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT), str(ROOT / 'tools')]
from core_modules.organize.task_artifacts import atomic_json, fingerprint, gallery_hashes


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TRANS_TASK_DIR'] = str(args.output / 'tasks')
    os.environ['TRANS_FEATURE_CACHE_DIR'] = str(args.output / 'features')
    import main
    import uvicorn
    from config.base import SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH, POSE_MODEL_PATH
    from core_modules.face_recognition.scrfd.scrfd_det import SCRFDDetector
    from core_modules.face_recognition.arcface.arcface_onnx import ArcFaceFeatureExtractor
    from core_modules.shot_classify import ShotTypeClassifier
    from core_modules.organize.task_versions import model_identity
    main.srv.face_recognition_model = SimpleNamespace(detector=SCRFDDetector(SCRFD_MODEL_PATH, device='cuda'),
        extractor=ArcFaceFeatureExtractor(ARCFACE_MODEL_PATH, device='cuda'))
    main.srv.pose_model = ShotTypeClassifier(POSE_MODEL_PATH)
    embedder = main._get_scene_embedder()
    atomic_json(args.output / 'model_contract.json', {
        'face': model_identity(main.srv.face_recognition_model), 'pose': model_identity(main.srv.pose_model),
        'embedding': model_identity(embedder), 'physical_gpu': args.gpu, 'remote_llm_enabled': False})
    uvicorn.run(main.app, host='127.0.0.1', port=args.port, lifespan='off', access_log=False, log_level='warning')


async def workflow(root, port, projects):
    import httpx
    from benchmark_face_task_http import ready
    from core_modules.organize.review_queue import load_scene_evidence
    results = []
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=30) as client:
        await ready(client)
        async def post(url, **kwargs):
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response.json()['data']

        async def completed(task_id):
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                response = await client.get('/api/organize/tasks/' + task_id)
                response.raise_for_status()
                row = response.json()['data']
                if row['status'] in ('completed', 'completed_with_errors', 'failed', 'cancelled'):
                    if row['status'] != 'completed':
                        atomic_json(root / 'failed_task.json', row)
                        raise RuntimeError('workflow_task_' + row['status'])
                    return row
                await asyncio.sleep(.25)
            raise TimeoutError('workflow_task_timeout')

        async def task(kind, params, key):
            body = {'kind': kind, 'params': params, 'idempotency_key': key}
            created = await post('/api/organize/tasks', json=body)
            duplicate = await post('/api/organize/tasks', json=body)
            if duplicate['task_id'] != created['task_id'] or duplicate['created']:
                raise AssertionError('idempotent_task_changed')
            return await completed(created['task_id'])

        for number, project in enumerate(projects):
            directory = Path(project['root'])
            params = {'input_dir': str(directory / 'photos'), 'knowledge_dir': str(directory / 'knowledge'),
                      'output_file': str(directory / 'evidence.jsonl'), 'use_embedder': True, 'use_llm': False}
            started = time.perf_counter()
            reason = await task('scene_reason', params, f'reason-{number}')
            review = str(directory / 'review')
            await post('/api/organize/scene/review/seed', data={'review_dir': review, 'evidence_file': params['output_file']})
            records = (await client.get('/api/organize/scene/review', params={'review_dir': review, 'limit': 200})).json()['data']['records']
            truth = {row['copy']: row for row in project['images']}
            # Both competing requests use an already confirmed photo-level label.
            records_by_image = {row['image']: row for row in records}
            first, second = [records_by_image[path] for path in project['review_images']]
            chosen = truth[first['image']]
            form = {'review_dir': review, 'image': first['image'], 'action': 'modify',
                    'scene_id': chosen['scene_id'], 'label': chosen['scene_label'],
                    'reviewer': 'acceptance-script-not-new-human-GT', 'expected_revision': first['review_revision']}
            conflicts = await asyncio.gather(client.post('/api/organize/scene/review', data=form),
                                            client.post('/api/organize/scene/review', data=form))
            statuses = sorted(response.status_code for response in conflicts)
            if statuses != [200, 409]:
                raise AssertionError('expected_one_review_conflict')
            await post('/api/organize/scene/review', data={'review_dir': review, 'image': second['image'], 'action': 'reject',
                        'reviewer': 'acceptance-script-not-new-human-GT', 'expected_revision': second['review_revision']})
            async def record(image):
                response = await client.get('/api/organize/scene/review', params={'review_dir': review, 'image': image})
                response.raise_for_status()
                return response.json()['data']['record']
            locked = [await record(row['image']) for row in (first, second)]
            exported = await post('/api/organize/scene/review/export', data={'review_dir': review})
            labels = json.loads(Path(exported['path']).read_text())
            if labels != {first['image']: chosen['scene_label']}:
                raise AssertionError('export_included_rejected_or_pending')
            # Fresh task creation exercises cache reuse and locked-review seeding.
            replay = await task('scene_reason', dict(params, output_file=str(directory / 'reanalysis.jsonl')), f'reason-replay-{number}')
            await post('/api/organize/scene/review/seed', data={'review_dir': review, 'evidence_file': str(directory / 'reanalysis.jsonl')})
            if [await record(row['image']) for row in (first, second)] != locked:
                raise AssertionError('reanalysis_overwrote_locked_review')
            organize = await task('organize', {'input_dir': params['input_dir'], 'output_dir': str(directory / 'organized'),
                'db_root': str(root / 'gallery'), 'keep_per_bucket': 2, 'gap_seconds': 300,
                'classify_role': True, 'identity_registry_file': str(root / 'identity_registry.json'),
                'scene_evidence_file': str(directory / 'review/review_queue.jsonl')}, f'organize-{number}')
            report = json.loads(Path(organize['result']['report_path']).read_text())
            artifacts = []
            for sequence, row in enumerate(report['records'], 1):
                source = truth[row['image']]
                best = directory / 'organized/best' / Path(row['image']).name
                output = directory / 'organized/organized' / row['organized_name']
                metadata = source['source_metadata']
                prefix = f"{sequence:02d}-{metadata['date']}-{metadata['venue']}-{metadata['play']}{metadata.get('event') or ''} "
                prefix = re.sub(r'[\\/:*?"<>|]', '', prefix)
                scene = chosen['scene_label'] if row['image'] == first['image'] else None
                remainder = row['organized_name'][len(prefix):] if row['organized_name'].startswith(prefix) else ''
                artifacts.append({'source': row['image'], 'output': str(output),
                    'content_equal': fingerprint(best) == fingerprint(output) == source['sha256'],
                    'metadata_equal': all(row.get(key) == metadata.get(key) for key in ('date', 'venue', 'play', 'event')),
                    'name_prefix_valid': row['organized_name'].startswith(prefix),
                    'scene_name_valid': remainder.startswith(re.sub(r'[\\/:*?"<>|]', '', scene)) if scene else bool(re.match(r'^scene-\d+(?:-|\.)', remainder)),
                    'safe_basename': output.name == row['organized_name'] and not re.search(r'[\\/:*?"<>|\x00-\x1f]', row['organized_name'])})
            evidence = load_scene_evidence(str(directory / 'review/review_queue.jsonl'))
            published = {row['source'] for row in artifacts}
            checks = {'reason_count': reason['result']['n_images'] == len(project['images']),
                      'reason_real_visual_retrieval': reason['result']['n_visual'] == len(project['images']),
                      'input_conservation': report['total_input'] == report['kept'] + report['discarded'] + report['failed'],
                      'copy_count': report['copied'] == len(artifacts) == len(list((directory / 'organized/organized').iterdir())),
                      'output_bytes_equal_sources': all(row['content_equal'] for row in artifacts),
                      'source_paths_unique': len({row['source'] for row in artifacts}) == len(artifacts),
                      'output_names_unique': len({row['output'] for row in artifacts}) == len(artifacts),
                      'raw_date_venue_play_event_preserved': all(row['metadata_equal'] for row in artifacts),
                      'output_filename_format': all(row['name_prefix_valid'] and row['safe_basename'] for row in artifacts),
                      'approved_label_and_unreviewed_placeholders': all(row['scene_name_valid'] for row in artifacts),
                      'approved_and_rejected_probes_both_published': {first['image'], second['image']} <= published,
                      'locked_records_survived_reanalysis': True, 'concurrent_review_one_success_one_conflict': True,
                      'export_only_approved': len(labels) == 1,
                      'rejected_not_used_as_label': second['image'] not in evidence.get('by_path', {}),
                      'no_failed_images': report['failed'] == 0}
            result = {'play_id': project['play_id'], 'checks': checks, 'input_count': len(project['images']),
                      'reason_task': reason['task_id'], 'replay_task': replay['task_id'], 'organize_task': organize['task_id'],
                      'kept': report['kept'], 'discarded': report['discarded'], 'copied': report['copied'],
                      'failed': report['failed'], 'scripted_reviews': 2, 'artifact_audit': artifacts,
                      'review_images': project['review_images'],
                      'elapsed_seconds': time.perf_counter() - started}
            results.append(result)
            atomic_json(directory / 'workflow_result.json', result)
            atomic_json(root / 'partial_results.json', results)
            print(json.dumps({'play': number + 1, 'input_count': len(project['images']),
                              'kept': result['kept'], 'checks': checks}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--gallery-root', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--input-kind', choices=('raw', 'organized'), default='raw',
                        help='Select filename scope explicitly; raw runs reject organized copies.')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if args.max_images < 8:
        raise ValueError('at_least_eight_workflow_images')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('new_empty_output_required')
    manifest = json.loads(args.manifest.read_text())
    groups = defaultdict(list)
    for row in manifest['rows']:
        groups[row['play_id']].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row['image_id'])
    selected = []
    for index in range(max(map(len, groups.values()))):
        selected.extend(groups[play][index] for play in sorted(groups) if index < len(groups[play]))
    selected = selected[:args.max_images]
    if len(selected) != args.max_images or len({row['content_hash'] for row in selected}) != len(selected):
        raise ValueError('not_enough_unique_workflow_images')
    from core_modules.organize.filename_parser import parse_filename, is_organized_filename
    from core_modules.organize.burst_dedup import normalize_group_key
    source_metadata = {row['source']: parse_filename(row['source']) for row in selected}
    if any(not all(info.get(key) for key in ('date', 'venue', 'play')) for info in source_metadata.values()):
        raise ValueError('filenames_with_date_venue_play_required')
    if any(is_organized_filename(row['source']) != (args.input_kind == 'organized') for row in selected):
        raise ValueError('filename_input_kind_mismatch')
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row['play_id'], normalize_group_key(row['source']))].append(row['source'])
    review_sources = {play: sorted(source for (pid, _), members in grouped.items()
                                 if pid == play and len(members) <= 2 for source in members)[:2]
                      for play in groups}
    if any(len(sources) != 2 for sources in review_sources.values()):
        raise ValueError('two_review_probes_in_groups_of_at_most_two_required_per_play')
    args.output.mkdir(parents=True)
    gallery_before = gallery_hashes(args.gallery_root)
    shutil.copytree(args.gallery_root, args.output / 'gallery')
    shutil.copy2(args.registry, args.output / 'identity_registry.json')
    projects = []
    for number, play in enumerate(sorted(groups)):
        directory = args.output / f'project_{number}'
        (directory / 'photos').mkdir(parents=True)
        (directory / 'knowledge').mkdir()
        scenes = [row for row in manifest['scenes'] if row['play_id'] == play]
        atomic_json(directory / 'knowledge/entities.json', {'scenes': scenes, 'roles': [], 'role_aliases': {}})
        images = []
        for row in selected:
            if row['play_id'] != play:
                continue
            source = (args.source_root / row['source']).resolve()
            if args.source_root.resolve() not in source.parents or fingerprint(source) != row['content_hash']:
                raise ValueError('source_scope_or_content_changed')
            destination = directory / 'photos' / source.name
            if destination.exists():
                raise ValueError('duplicate_workflow_basename')
            shutil.copy2(source, destination)
            images.append({'source': str(source), 'copy': str(destination), 'sha256': row['content_hash'],
                           'scene_id': row['scene_id'], 'scene_label': row['scene_label'],
                           'source_metadata': source_metadata[row['source']]})
        review_images = [str(directory / 'photos' / Path(source).name) for source in review_sources[play]]
        projects.append({'play_id': play, 'root': str(directory), 'images': images, 'review_images': review_images})
    atomic_json(args.output / 'input_manifest.json', {'projects': projects, 'source_manifest_sha256': fingerprint(args.manifest),
                'gallery_before': gallery_before, 'registry_sha256': fingerprint(args.registry)})
    files = ['app/main.py', 'app/organize_tasks.py', 'core_modules/organize/task_versions.py',
             'core_modules/organize/task_queue.py', 'core_modules/organize/scene_reasoner.py',
             'core_modules/organize/organize_task.py', 'core_modules/organize/smart_organizer.py',
             'core_modules/organize/filename_parser.py', 'core_modules/organize/burst_dedup.py',
             'core_modules/organize/review_queue.py', 'core_modules/organize/review_store.py',
             'tools/check_real_workflow.py']
    state = {'status': 'running', 'command': [sys.executable, *sys.argv],
             'started_at': datetime.now(timezone.utc).isoformat(),
             'source_sha256': {name: fingerprint(ROOT / name) for name in files}}
    atomic_json(args.output / 'run_state.json', state)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--port', str(port)]
    with (args.output / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        state.update(server_pid=process.pid, port=port)
        atomic_json(args.output / 'run_state.json', state)
        try:
            results = asyncio.run(workflow(args.output, port, projects))
        except Exception as exc:
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            atomic_json(args.output / 'run_state.json', state)
            raise
        finally:
            process.terminate()
            process.wait(timeout=20)
    checks = {'all_project_checks': all(all(row['checks'].values()) for row in results),
              'originals_unchanged': all(fingerprint(row['source']) == fingerprint(row['copy']) == row['sha256'] for project in projects for row in project['images']),
              'official_gallery_unchanged': gallery_hashes(args.gallery_root) == gallery_before,
              'gallery_copy_unchanged': gallery_hashes(args.output / 'gallery') == gallery_before,
              'registry_unchanged': fingerprint(args.registry) == fingerprint(args.output / 'identity_registry.json'),
              'measured_code_unchanged': all(fingerprint(ROOT / name) == digest for name, digest in state['source_sha256'].items())}
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks, 'projects': results,
               'input_count': len(selected), 'project_count': len(projects), 'physical_gpu': args.gpu,
               'input_kind': args.input_kind + '_photo_filenames',
               'formal_u8_gate': 'not_passed', 'quality_gate': 'insufficient_data',
               'limitations': ['Functional replay of actual SigLIP2, SCRFD/ArcFace, pose and role paths, with use_llm=false.',
                              'Scripted review decisions in isolated copies are not new human annotations.',
                              'Existing gallery/query overlap is retained only for workflow testing, not identity accuracy.',
                              'One run; no controlled U2 throughput comparison or final-test quality claim.',
                              'Full production startup is disabled; only the models needed for this workflow are loaded.']}
    atomic_json(args.output / 'summary.json', summary)
    state.update(status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output / 'run_state.json', state)
    (args.output / 'report.md').write_text('\n'.join(['# Real workflow acceptance', '',
        f"Functional checks: {summary['status']}; {len(selected)} images across {len(projects)} projects.",
        'Formal U8 remains not_passed; U0/U2 and final-test quality requirements are incomplete.', '',
        *summary['limitations'], '']), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'input_count': len(selected), 'checks': checks}), flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
