"""Persistent scene analysis API; business functions are shared by old routes."""
import json
import os
from pathlib import Path
import threading

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from core_modules.organize.review_store import atomic_rows
from core_modules.organize.task_queue import TaskQueue, TaskConflict, TaskQueueFull, TaskUnavailable, BUSINESS_LOCK
from core_modules.tools.image_io import list_images
from core_modules.organize.task_artifacts import fingerprint, gallery_hashes, file_state, face_database_images

router = APIRouter(prefix='/api/organize/tasks')
TASK_ROOT = Path(os.environ.get('TRANS_TASK_DIR', 'outputs/organize_tasks'))
_manager = None
_manager_lock = threading.Lock()


def prepare_face_database(params, defer_hash=False):
    if set(params) - {'organized_dirs', 'db_root', 'max_anchors', 'dry_run', 'output_dir'}:
        raise ValueError('人脸建库参数含未知字段')
    dirs = params.get('organized_dirs')
    if isinstance(dirs, str):
        dirs = [value.strip() for value in dirs.split(',') if value.strip()]
    if not isinstance(dirs, list) or not dirs or any(not isinstance(path, str) or not os.path.isdir(path) for path in dirs):
        raise ValueError('organized_dirs 必须包含现有照片目录')
    if not isinstance(params.get('db_root'), str) or not params['db_root']:
        raise ValueError('db_root 必须为人脸库目录')
    if type(params.get('dry_run', False)) is not bool:
        raise ValueError('dry_run 必须为布尔值')
    maximum = params.get('max_anchors', 8)
    if type(maximum) is not int or not 1 <= maximum <= 1000:
        raise ValueError('max_anchors 必须为 1–1000 的整数')
    root = Path(params['db_root']).resolve()
    dirs = list(dict.fromkeys(str(Path(path).resolve()) for path in dirs))
    if any(root == Path(path) or root in Path(path).parents or Path(path) in root.parents for path in dirs):
        raise ValueError('人脸库必须位于照片目录之外')
    output = params.get('output_dir')
    if output is not None:
        if not isinstance(output, str) or not output:
            raise ValueError('output_dir 必须为独立报告目录')
        output = str(Path(output).resolve())
        if any(Path(output) == Path(path) or Path(output) in Path(path).parents or Path(path) in Path(output).parents
               for path in dirs + [str(root)]):
            raise ValueError('建库报告必须位于原图和人脸库目录之外')
    if root.exists() and not root.is_dir():
        raise ValueError('db_root 必须为目录')
    items, counts, gallery = [], {}, []
    if root.exists():
        for path in face_database_images(root):
            relative = path.relative_to(root)
            gallery.append(str(path))
            if len(relative.parts) == 2:
                person = relative.parts[0]
                if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    counts[person] = counts.get(person, 0) + 1
                items.append({'image': str(path), 'stage': 'gallery', 'person': person})
            else:
                # Legacy matching reads only direct actor images. Still bind
                # other images to the manifest so changes are detected.
                items.append({'image': str(path), 'stage': 'gallery_snapshot'})
    photos = []
    for directory in dirs:
        for name in list_images(directory):
            path = str(Path(directory, name).resolve())
            if Path(directory) not in Path(path).parents:
                raise ValueError('照片符号链接不能指向输入目录之外')
            if path not in photos:
                photos.append(path)
                items.append({'image': path, 'stage': 'anchors', 'done_style': '整理完成' in directory})
    if not photos:
        raise ValueError('照片目录中没有图片')
    for item in items:
        item.update({'file_state': file_state(item['image'])} if defer_hash else {'sha256': fingerprint(item['image'])})
    return dict(params, organized_dirs=dirs, db_root=str(root), max_anchors=maximum,
                dry_run=params.get('dry_run', False), output_dir=output, photo_paths=photos,
                gallery_paths=gallery, gallery_counts=counts,
                input_hashes={item['image']: item.get('sha256') for item in items}), items


def face_database_handler(params):
    import main
    from core_modules.organize.face_database_task import FaceDatabaseTask
    if main.srv.face_recognition_model is None:
        raise ValueError('Face recognition model not initialized')
    return FaceDatabaseTask(params, main.srv.face_recognition_model)


def prepare_face_cluster(params, defer_hash=False):
    if set(params) - {'input_dir', 'output_dir', 'dist_threshold', 'min_face_area',
                      'clustering_backend', 'graph_config', 'cannot_link_exemptions'}:
        raise ValueError('聚类参数含未知字段')
    if not isinstance(params.get('input_dir'), str) or not os.path.isdir(params['input_dir']):
        raise ValueError('input_dir 必须为现有目录')
    if not isinstance(params.get('output_dir'), str) or not params['output_dir']:
        raise ValueError('output_dir 必须为独立输出目录')
    threshold, area = params.get('dist_threshold', .5), params.get('min_face_area', 1600)
    if type(threshold) not in (int, float) or not 0 < threshold <= 2:
        raise ValueError('dist_threshold 必须在 (0, 2]')
    if type(area) is not int or not 1 <= area <= 100000000:
        raise ValueError('min_face_area 必须为正整数')
    backend = params.get('clustering_backend', 'legacy')
    if backend not in ('legacy', 'graph_experimental'):
        raise ValueError('invalid_face_clustering_backend')
    from dataclasses import asdict
    from core_modules.organize.face_graph_cluster import FaceGraphConfig
    options = params.get('graph_config') or {}
    if not isinstance(options, dict) or set(options) - set(FaceGraphConfig.__dataclass_fields__):
        raise ValueError('invalid_face_graph_config')
    graph_config = asdict(FaceGraphConfig(**options))
    source, output = Path(params['input_dir']).resolve(), Path(params['output_dir']).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('聚类输出必须位于原图目录之外')
    images = [str((source / name).resolve()) for name in list_images(source)]
    if not images:
        raise ValueError('输入目录中没有图片')
    exemptions = params.get('cannot_link_exemptions', [])
    if not isinstance(exemptions, list) or any(not isinstance(row, dict) or set(row) != {'image', 'face_index'} or
                                             row['image'] not in images or type(row['face_index']) is not int or
                                             row['face_index'] < 0 for row in exemptions):
        raise ValueError('invalid_cannot_link_exemptions')
    items = [{'image': image, 'stage': 'faces',
              **({'file_state': file_state(image)} if defer_hash else {'sha256': fingerprint(image)})} for image in images]
    return dict(params, input_dir=str(source), output_dir=str(output), dist_threshold=threshold,
                min_face_area=area, clustering_backend=backend, graph_config=graph_config,
                cannot_link_exemptions=exemptions,
                input_hashes={item['image']: item.get('sha256') for item in items}), items


def face_cluster_handler(params):
    import main
    from core_modules.organize.face_cluster_task import FaceClusterTask
    if main.srv.face_recognition_model is None:
        raise ValueError('Face recognition model not initialized')
    return FaceClusterTask(params, main.srv.face_recognition_model)


def prepare_organize(params, defer_hash=False):
    allowed = {'input_dir', 'output_dir', 'db_root', 'keep_per_bucket', 'gap_seconds',
               'classify_role', 'scene_labels_file', 'scene_evidence_file',
               'identity_registry_file', 'face_threshold', 'face_margin',
               'scene_strategy', 'scene_change_config', 'scene_order_trusted'}
    if set(params) - allowed:
        raise ValueError('整理参数含未知字段')
    for field in ('input_dir', 'db_root'):
        if not isinstance(params.get(field), str) or not os.path.isdir(params[field]):
            raise ValueError(f'{field} 必须为现有目录')
    for field, default, lower, upper in (('keep_per_bucket', 2, 1, 1000), ('gap_seconds', 300, 1, 86400)):
        if type(params.get(field, default)) is not int or not lower <= params.get(field, default) <= upper:
            raise ValueError(f'{field} 必须为 {lower}–{upper} 的整数')
    if type(params.get('classify_role', True)) is not bool:
        raise ValueError('classify_role 必须为布尔值')
    if params.get('scene_strategy', 'legacy') not in ('legacy', 'window_change_points'):
        raise ValueError('unknown_scene_split_strategy')
    if type(params.get('scene_order_trusted', False)) is not bool:
        raise ValueError('scene_order_trusted 必须为布尔值')
    from dataclasses import asdict
    from core_modules.organize.scene_change_points import ChangePointConfig
    options = params.get('scene_change_config') or {}
    if not isinstance(options, dict) or set(options) - set(ChangePointConfig.__dataclass_fields__):
        raise ValueError('invalid_scene_change_config')
    options = dict(options)
    options.setdefault('max_gap_seconds', params.get('gap_seconds', 300))
    change_config = asdict(ChangePointConfig(**options))
    import math
    from config.base import KNOWN_FACE_THRESHOLD, FACE_AMBIGUOUS_MARGIN
    for field, default, upper in (('face_threshold', KNOWN_FACE_THRESHOLD, 1),
                                  ('face_margin', FACE_AMBIGUOUS_MARGIN, 2)):
        value = params.get(field, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'{field} 必须在 [0, {upper}]')
    if params.get('identity_registry_file') is not None and not isinstance(params['identity_registry_file'], str):
        raise ValueError('identity_registry_file 必须为路径字符串')
    output = params.get('output_dir')
    if not isinstance(output, str) or not output:
        raise ValueError('output_dir 必须为独立输出目录')
    output = str(Path(output).resolve())
    source = str(Path(params['input_dir']).resolve())
    if output == source or Path(source).is_relative_to(Path(output)):
        raise ValueError('整理输出不可为原图目录或其父目录')
    images = [str(Path(source, name).resolve()) for name in list_images(source)]
    if not images:
        raise ValueError('输入目录中没有图片')
    items = [{'image': image, 'stage': 'quality',
              **({'file_state': file_state(image)} if defer_hash else {'sha256': fingerprint(image)})} for image in images]
    labels, evidence = None, None
    if params.get('scene_labels_file'):
        with open(params['scene_labels_file'], encoding='utf-8') as stream:
            labels = json.load(stream)
    if params.get('scene_evidence_file'):
        from core_modules.organize.review_queue import load_scene_evidence
        evidence = load_scene_evidence(params['scene_evidence_file'])
    prepared = dict(params, input_dir=source, output_dir=output,
                    db_root=str(Path(params['db_root']).resolve()),
                    keep_per_bucket=params.get('keep_per_bucket', 2), gap_seconds=params.get('gap_seconds', 300),
                    classify_role=params.get('classify_role', True), scene_labels=labels, scene_evidence=evidence,
                    scene_strategy=params.get('scene_strategy', 'legacy'), scene_change_config=change_config,
                    scene_order_trusted=params.get('scene_order_trusted', False),
                    input_hashes={item['image']: item.get('sha256') for item in items})
    # Gallery changes must not silently mix identity decisions across a resumed task.
    prepared['gallery_hashes'] = gallery_hashes(prepared['db_root'])
    from core_modules.organize.identity_registry import registry_path, registry_digest, IdentityRegistry
    identity_file = registry_path(prepared['db_root'], params.get('identity_registry_file'))
    identity_hash = registry_digest(identity_file)
    IdentityRegistry.load(identity_file)
    if registry_digest(identity_file) != identity_hash:
        raise ValueError('identity_registry_changed_during_submission')
    prepared.update(identity_registry_file=str(identity_file), identity_registry_sha256=identity_hash,
                    face_threshold=params.get('face_threshold', KNOWN_FACE_THRESHOLD),
                    face_margin=params.get('face_margin', FACE_AMBIGUOUS_MARGIN))
    return prepared, items


def organize_handler(params):
    import main
    from core_modules.organize.organize_task import OrganizeTask
    from core_modules.organize.smart_organizer import SmartOrganizer
    if main.srv.face_recognition_model is None:
        raise ValueError('Face recognition model not initialized')
    if gallery_hashes(params['db_root']) != params['gallery_hashes']:
        raise ValueError('face_gallery_content_changed; 请创建新任务')
    from core_modules.organize.identity_registry import registry_digest
    if registry_digest(params['identity_registry_file']) != params['identity_registry_sha256']:
        raise ValueError('identity_registry_content_changed; 请创建新任务')
    role = main._get_organize_models(True)[1] if params['classify_role'] else None
    organizer = SmartOrganizer(main.srv.face_recognition_model, params['db_root'],
                               shot_classifier=main.srv.pose_model, embedder=main._get_scene_embedder(),
                               role_classifier=role, feature_cache_dir=main.FEATURE_CACHE_DIR,
                               identity_registry_file=params['identity_registry_file'] if params['identity_registry_sha256'] is not None else None,
                               face_threshold=params['face_threshold'], face_margin=params['face_margin'])
    return OrganizeTask(params, organizer)


def prepare_knowledge(params, defer_hash=False):
    if set(params) - {'info_path', 'play_root', 'output_dir'}:
        raise ValueError('知识库参数含未知字段')
    for field in ('info_path', 'play_root', 'output_dir'):
        if not isinstance(params.get(field), str) or not params[field]:
            raise ValueError(f'{field} 必须为路径字符串')
    if not os.path.isfile(params['info_path']) or not os.path.isdir(params['play_root']):
        raise ValueError('剧情文件或剧目目录不存在')
    from core_modules.organize.play_source import load_local_sources, extract_urls
    from core_modules.organize.play_knowledge import _scenes_from_filenames
    info_hash = fingerprint(params['info_path'])
    sources = load_local_sources(params['info_path'], params['play_root'])
    sources['urls'] = extract_urls(params['info_path'])
    items = [{'image': str(Path(image).resolve()), 'stage': 'media',
              **({'file_state': file_state(image)} if defer_hash else {'sha256': fingerprint(image)})}
             for image in sources['image_paths']]
    prepared = dict(params, output_dir=str(Path(params['output_dir']).resolve()),
                    info_path=str(Path(params['info_path']).resolve()), play_root=str(Path(params['play_root']).resolve()),
                    sources=sources, filename_scenes=_scenes_from_filenames(params['play_root']),
                    info_sha256=info_hash, input_hashes={item['image']: item.get('sha256') for item in items})
    if fingerprint(params['info_path']) != info_hash:
        raise ValueError('plot_content_changed_during_submission')
    return prepared, items


def knowledge_handler(params):
    from core_modules.organize.knowledge_task import KnowledgeTask
    return KnowledgeTask(params)


def execute_staged(kind, params):
    """Legacy sync routes share task handlers without starting a second worker."""
    from core_modules.organize.task_queue import encode
    preparer, factory = {'organize': (prepare_organize, organize_handler),
                         'knowledge': (prepare_knowledge, knowledge_handler),
                         'face_database': (prepare_face_database, face_database_handler)}[kind]
    prepared, pending = preparer(params)
    class InlineContext:
        def __init__(self):
            self.rows = []
            self.plans = {}
        def check(self):
            pass
        def checkpoint(self, key):
            return self.plans.get(key)
        def results(self):
            return self.rows
        def result_for(self, image, stage):
            return next(row for row in self.rows if row['image'] == image and row.get('stage') == stage and not row.get('error'))
        def advance(self, key, value, items):
            if key in self.plans:
                return False
            self.plans[key] = value
            pending.extend(items)
            return True
    with BUSINESS_LOCK:
        handler = factory(prepared)
        context = InlineContext()
        handler.context = context
        while True:
            for item in pending:
                try:
                    row = handler.process(item)
                    # Match the persisted queue's JSON validation and round trip.
                    row = json.loads(encode(row))
                except Exception as exc:
                    row = {'image': item['image'], 'stage': item['stage'],
                           'error': {'type': type(exc).__name__, 'message': str(exc)}}
                context.rows.append(row)
            pending.clear()
            if not handler.advance():
                return handler.finish(context.rows)


def prepare_scene(params, defer_hash=False):
    allowed = {'input_dir', 'knowledge_dir', 'output_file', 'top_k', 'use_llm', 'use_embedder',
               'visual_policy', 'visual_cache_dir', 'ocr_images', 'ocr_config'}
    if set(params) - allowed:
        raise ValueError('分析参数含未知字段')
    for field in ('input_dir', 'knowledge_dir'):
        if not isinstance(params.get(field), str) or not os.path.isdir(params[field]):
            raise ValueError(f'{field} 必须为现有目录')
    top_k = params.get('top_k', 5)
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k 必须为 1–100 的整数')
    for field in ('use_llm', 'use_embedder'):
        if type(params.get(field, False)) is not bool:
            raise ValueError(f'{field} 必须为布尔值')
    if not isinstance(params.get('output_file', ''), str):
        raise ValueError('output_file 必须为路径字符串')
    from dataclasses import asdict
    from core_modules.organize.visual_evidence import VisualPolicy
    visual_policy = asdict(VisualPolicy.from_dict(params.get('visual_policy')))
    if params.get('visual_cache_dir') is not None and (not isinstance(params['visual_cache_dir'], str) or not params['visual_cache_dir']):
        raise ValueError('visual_cache_dir 必须为目录路径字符串')
    from core_modules.organize.play_knowledge import read_knowledge
    entities = read_knowledge(params['knowledge_dir'])['entities']
    knowledge = {key: entities.get(key, {} if key == 'role_aliases' else [])
                 for key in ('scenes', 'roles', 'role_aliases')}
    images = [str(Path(params['input_dir'], name).resolve()) for name in list_images(params['input_dir'])]
    if not images:
        raise ValueError('输入目录中没有图片')
    ocr_images = params.get('ocr_images', [])
    if not isinstance(ocr_images, list) or len(ocr_images) > 64 or any(image not in images for image in ocr_images):
        raise ValueError('ocr_images 必须是当前输入中的最多64张明确含字图片')
    ocr_config = params.get('ocr_config')
    if ocr_config is not None:
        if not isinstance(ocr_config, dict) or set(ocr_config) != {'python', 'detector_dir', 'recognizer_dir', 'cache_dir'} or any(
                not isinstance(value, str) or not value for value in ocr_config.values()):
            raise ValueError('invalid_ocr_config')
    if ocr_images and not ocr_config:
        raise ValueError('ocr_images 需要本地ocr_config')
    output = str(Path(params.get('output_file') or Path(params['input_dir']) / 'scene_evidence.jsonl').resolve())
    if output in images or not output.endswith('.jsonl'):
        raise ValueError('证据输出必须为独立的 .jsonl 文件')
    # Persist the exact knowledge snapshot and content identity used by this task.
    items = [{'image': image, **({'file_state': file_state(image)} if defer_hash else {'sha256': fingerprint(image)})} for image in images]
    prepared = dict(params, knowledge=knowledge, output_file=output, top_k=top_k,
                    visual_policy=visual_policy,
                    input_hashes={item['image']: item.get('sha256') for item in items})
    return prepared, items


def scene_handler(params):
    from core_modules.organize.scene_reasoner import SceneReasoner
    client = vision_client = embedder = None
    if params.get('use_llm'):
        from core_modules.organize.llm_client import client_from_env, vision_client_from_env
        vision_client = vision_client_from_env()
        client = vision_client or client_from_env()
    if params.get('use_embedder'):
        # main may be imported as top-level by the legacy entry point.
        import main
        from core_modules.organize.multimodal_evidence import MultimodalRetriever
        embedder = MultimodalRetriever(embedder=main._get_scene_embedder(), cache_dir=main.FEATURE_CACHE_DIR)
    from core_modules.organize.visual_evidence import VisualEvidenceController, VisualPolicy
    from core_modules.organize.feature_cache import signature
    from config.base import VISION_IMAGE_MAX_SIDE
    policy = VisualPolicy.from_dict(params.get('visual_policy'))
    controller = None
    if vision_client is not None:
        scope = signature([params['output_file'], params['input_hashes'], params['knowledge']])
        controller = VisualEvidenceController(
            vision_client, params.get('visual_cache_dir') or 'outputs/visual_evidence_cache',
            str(params['output_file']) + '.visual_budget.sqlite3', scope,
            len(params['input_hashes']), policy, max_side=VISION_IMAGE_MAX_SIDE)
    reasoner = SceneReasoner(params['knowledge'], client=client, embedder=embedder,
                             vision_client=vision_client, top_k=params.get('top_k', 5),
                             visual_controller=controller)
    ocr = None
    if params.get('ocr_images'):
        from core_modules.organize.ocr_evidence import PaddleOCRProvider
        ocr = PaddleOCRProvider(**params['ocr_config'])
        reasoner.ocr_provider = ocr
        reasoner.ocr_images = set(params['ocr_images'])
    # A configured visual pipeline must not silently switch to an unbudgeted
    # language-only model if the vision client is unavailable.
    if params.get('use_llm') and vision_client is None:
        reasoner.client = None

    def process(item):
        path = item['image']
        if fingerprint(path) != item['sha256']:
            raise ValueError('source_content_changed')
        row = reasoner.reason(path)
        if fingerprint(path) != item['sha256']:
            raise ValueError('source_content_changed_during_analysis')
        return row

    def finish(rows):
        for row in rows:
            if not row.get('error') and fingerprint(row['image']) != params['input_hashes'][row['image']]:
                raise ValueError('completed_source_content_changed; 请创建新的分析任务。')
        path = Path(params['output_file'])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_rows(path, rows)
        if embedder is not None:
            embedder.save()
        return {'n_images': len(rows), 'output_file': str(path),
                'n_unknown': sum(row['label'] == 'unknown' for row in rows),
                'n_failed': sum(bool(row.get('error')) for row in rows),
                'labels': [row['label'] for row in rows],
                'n_visual': sum(row.get('retrieval_mode') == 'visual' for row in rows),
                'llm': reasoner.client is not None, 'visual': embedder is not None,
                'vision_budget': controller.budget.summary() if controller else None,
                'n_observed': sum(any(e.get('type') == 'observation' for e in row.get('evidence', [])) for row in rows),
                'n_vision_cache_hits': sum(bool(row.get('model_trace', {}).get('vision', {}).get('cache_hit')) for row in rows)}

    from core_modules.organize.task_versions import VersionedHandler, execution_code, model_identity, client_identity
    from config.base import VISION_IMAGE_MAX_SIDE, VISION_ENABLE_THINKING
    code_version = execution_code([
        'app/organize_tasks.py', 'core_modules/organize/scene_reasoner.py',
        'core_modules/organize/visual_evidence.py',
        'core_modules/organize/ocr_evidence.py', 'tools/paddle_ocr_worker.py',
        'core_modules/organize/multimodal_evidence.py', 'core_modules/organize/llm_client.py',
        'core_modules/image_search.py', 'core_modules/organize/embedding_gemma.py',
        'core_modules/organize/feature_cache.py', 'core_modules/organize/task_artifacts.py'])
    def contract():
        return {'code': code_version,
                'embedding': model_identity(embedder.embedder) if embedder else None,
                'client': client_identity(client), 'vision_client': client_identity(vision_client),
                'vision_settings': [VISION_IMAGE_MAX_SIDE, VISION_ENABLE_THINKING],
                'visual_policy': params.get('visual_policy', {}),
                'ocr_identity': ocr.identity if ocr else None,
                'remote_weights_verified': getattr(vision_client, 'verified_weights_sha256', None) is not None}

    class SceneTask:
        context = None
        model_contract = staticmethod(contract)

        def process(self, item):
            reasoner.boundary_check = self.context.check if self.context else None
            return process(item)

        def advance(self):
            return False

        def finish(self, rows):
            return finish(rows)

        def __iter__(self):
            return iter((self.process, self.finish))

    return SceneTask()


def execute_scene(params):
    """Same scene executor for legacy synchronous API and agent tools."""
    prepared, items = prepare_scene(params)
    with BUSINESS_LOCK:
        process, finish = scene_handler(prepared)
        rows = []
        for item in items:
            try:
                rows.append(process(item))
            except Exception as exc:
                rows.append({'image': item['image'], 'label': 'unknown', 'scene_id': 'unknown',
                             'decision': 'error', 'confidence': None,
                             'review': {'status': 'pending', 'reviewer': None, 'updated_at': None},
                             'error': {'type': type(exc).__name__, 'message': str(exc)}})
        return finish(rows)


def deferred_handler(kind, params):
    if '_manifest_items' not in params:
        return {'scene_reason': scene_handler, 'organize': organize_handler, 'knowledge': knowledge_handler,
                'face_cluster': face_cluster_handler, 'face_database': face_database_handler}[kind](params)
    return PreparedTask(kind, params)


class PreparedTask:
    """Hash manifest items off the request thread, then atomically add work."""
    def __init__(self, kind, params):
        self.kind, self.params = kind, params
        self.context = None
        self.delegate = None

    def model_contract(self):
        if self.context is None or self.context.checkpoint('input_fingerprints') is None:
            return None
        contract = getattr(self._delegate(), 'model_contract', None)
        return contract() if contract else None

    def _rows(self):
        rows = []
        stages = {'organize': 'quality', 'knowledge': 'media', 'scene_reason': None,
                  'face_cluster': 'faces', 'face_database': 'anchors'}
        for row in self.context.results():
            if row.get('stage') != 'fingerprint':
                rows.append(row)
            elif row.get('error'):
                rows.append(dict(row, stage=stages[self.kind]))
        return rows

    def _delegate(self):
        if self.delegate is not None:
            return self.delegate
        prepared = dict(self.params)
        plan = self.context.checkpoint('input_fingerprints')
        if plan is None:
            raise ValueError('input manifest not committed')
        prepared['input_hashes'] = plan['hashes']
        factory = {'scene_reason': scene_handler, 'organize': organize_handler, 'knowledge': knowledge_handler,
                   'face_cluster': face_cluster_handler, 'face_database': face_database_handler}[self.kind]
        self.delegate = factory(prepared)
        if hasattr(self.delegate, 'advance'):
            outer = self
            class Context:
                check = self.context.check
                checkpoint = self.context.checkpoint
                advance = self.context.advance
                result_for = self.context.result_for
                def results(self):
                    return outer._rows()
            self.delegate.context = Context()
        return self.delegate

    def process(self, item):
        if item.get('stage') == 'fingerprint':
            path = item['image']
            if file_state(path) != item['file_state']:
                raise ValueError('source_changed_before_manifest_hash')
            digest = fingerprint(path, self.context.check)
            if file_state(path) != item['file_state']:
                raise ValueError('source_changed_during_manifest_hash')
            return {'image': path, 'stage': 'fingerprint', 'sha256': digest}
        delegate = self._delegate()
        return delegate.process(item) if hasattr(delegate, 'advance') else delegate[0](item)

    def advance(self):
        if self.context.checkpoint('input_fingerprints') is None:
            rows = {row['image']: row for row in self.context.results()}
            hashes = {image: row.get('sha256') for image, row in rows.items()}
            items = [dict(item, sha256=hashes[item['image']]) for item in self.params['_manifest_items']
                     if hashes[item['image']] is not None]
            self.context.advance('input_fingerprints', {'hashes': hashes}, items)
            return True
        delegate = self._delegate()
        return delegate.advance() if hasattr(delegate, 'advance') else False

    def finish(self, rows):
        delegate = self._delegate()
        return delegate.finish(self._rows()) if hasattr(delegate, 'advance') else delegate[1](self._rows())


def manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TaskQueue(TASK_ROOT, {kind: (lambda params, kind=kind: deferred_handler(kind, params))
                                            for kind in ('scene_reason', 'organize', 'knowledge', 'face_cluster', 'face_database')},
                                 capacity=int(os.environ.get('TRANS_TASK_CAPACITY', '8')))
        return _manager


def shutdown():
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
            _manager = None


def response(operation, status=200):
    try:
        return JSONResponse(status_code=status, content={'code': 200, 'data': operation()})
    except KeyError as exc:
        return JSONResponse(status_code=404, content={'message': str(exc)})
    except TaskConflict as exc:
        return JSONResponse(status_code=409, content={'message': str(exc)})
    except (TaskQueueFull, TaskUnavailable) as exc:
        return JSONResponse(status_code=503, content={'message': str(exc)})
    except (ValueError, OSError) as exc:
        return JSONResponse(status_code=400, content={'message': str(exc)})


@router.post('')
def create_task(body: dict = Body(...)):
    def create():
        preparers = {'scene_reason': prepare_scene, 'organize': prepare_organize, 'knowledge': prepare_knowledge,
                     'face_cluster': prepare_face_cluster, 'face_database': prepare_face_database}
        kind = body.get('kind')
        if set(body) - {'kind', 'params', 'idempotency_key'} or kind not in preparers:
            raise ValueError('后台任务支持 scene_reason、organize、knowledge、face_cluster、face_database')
        params = body.get('params')
        if not isinstance(params, dict):
            raise ValueError('params 必须为对象')
        key = body.get('idempotency_key')
        if key is not None and not isinstance(key, str):
            raise ValueError('idempotency_key 必须为字符串')
        prepared, items = preparers[kind](params, defer_hash=True)
        prepared['_manifest_items'] = items
        items = [dict(item, stage='fingerprint') for item in items]
        output = ([prepared['db_root']] + ([prepared['output_dir']] if prepared.get('output_dir') else [])) \
            if kind == 'face_database' else prepared.get('output_file') or prepared['output_dir']
        task, created = manager().submit(kind, prepared, items, key, output)
        return dict(task, created=created)
    return response(create, 202)


@router.get('')
def list_tasks(limit: int = 50):
    return response(lambda: {'tasks': manager().list(limit)})


@router.get('/{task_id}')
def get_task(task_id: str):
    return response(lambda: manager().get(task_id))


@router.post('/{task_id}/cancel')
def cancel_task(task_id: str):
    return response(lambda: manager().cancel(task_id))


@router.get('/{task_id}/items')
def task_items(task_id: str, limit: int = 50, offset: int = 0):
    return response(lambda: {'items': manager().item_page(task_id, limit, offset)})


@router.post('/{task_id}/resume')
def resume_task(task_id: str):
    return response(lambda: manager().resume(task_id), 202)
