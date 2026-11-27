"""独立于 Web/GPU 服务的剧照工具；重型模块按需加载。"""
import json
import shutil
import uuid
from pathlib import Path

from core_modules.organize.agent_tools import IMAGE_EXTENSIONS, register_image_tools


def _directory(path):
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'目录不存在: {source}')
    return source


def _fresh_output(path, source=None):
    target = Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ValueError('输出路径已存在，请指定新路径；禁止覆盖')
    if source and target.resolve().is_relative_to(Path(source).expanduser().resolve()):
        raise ValueError('输出必须在输入目录之外')
    return target


def inspect_photos(input_dir, limit=20):
    from core_modules.organize.filename_parser import parse_dirname, parse_filename
    source = _directory(input_dir)
    images = sorted(path for path in source.iterdir() if path.is_file()
                    and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return {'input_dir': str(source), 'n_images': len(images),
            'directory_metadata': parse_dirname(source.name),
            'samples': [{'image': str(path), 'metadata': parse_filename(path.name)}
                        for path in images[:limit]],
            'evidence_type': 'filename_only', 'auto_approved': True}


def group_photos_by_actor(input_dir, workspace_output_dir):
    from core_modules.organize.filename_parser import parse_filename
    source = _directory(input_dir)
    workspace = Path(workspace_output_dir).expanduser().resolve()
    if workspace.is_relative_to(source):
        raise ValueError('项目输出必须位于剧照目录之外')
    images = sorted(path for path in source.iterdir() if not path.is_symlink() and path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError('输入目录没有支持的图片')
    groups = {}
    unlabelled = []
    for image in images:
        actors = {actor for actor, role in parse_filename(image.name)['people'] if actor and role}
        if not actors:
            unlabelled.append(image.name)
        for actor in sorted(actors):
            groups.setdefault(actor, []).append(image.name)
    summary = {'evidence_type': 'filename_only', 'n_images': len(images),
               'n_grouped': len(images) - len(unlabelled), 'n_actors': len(groups),
               'n_unlabelled': len(unlabelled), 'requires_human_review': bool(unlabelled)}
    if not groups:
        return summary
    target = workspace / ('actor-albums-' + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    report = target / 'actor_groups.json'
    try:
        for actor, names in groups.items():
            album = target / actor
            album.mkdir()
            for name in names:
                shutil.copy2(source / name, album / name)
        report.write_text(json.dumps(dict(summary, actors=groups, unlabelled=unlabelled),
                                     ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        shutil.rmtree(target)
        raise
    return dict(summary, output_dir=str(target), report_path=str(report))


def knowledge_summary(knowledge_dir):
    root = _directory(knowledge_dir)
    from core_modules.organize.play_knowledge import read_knowledge
    data = read_knowledge(root)['entities']
    return {'knowledge_dir': str(root), 'scenes': [scene.get('label') for scene in data.get('scenes', [])],
            'roles': data.get('roles', []), 'role_aliases': data.get('role_aliases', {}),
            'auto_approved': True}


def build_play_knowledge(info_path, play_root, output_dir):
    from core_modules.organize.play_knowledge import build_knowledge
    source = _directory(play_root)
    target = _fresh_output(output_dir, source)
    knowledge = build_knowledge(str(Path(info_path).expanduser()), str(source), str(target))
    return {'knowledge_dir': str(target), 'stats': knowledge['stats']}


def scene_reason(input_dir, knowledge_dir, output_file, top_k=5):
    from core_modules.organize.scene_reasoner import reason_batch
    source = _directory(input_dir)
    target = _fresh_output(output_file, source)
    if target.suffix.lower() != '.jsonl':
        raise ValueError('output_file 必须是 JSONL 文件')
    from core_modules.organize.play_knowledge import read_knowledge
    knowledge = read_knowledge(_directory(knowledge_dir))['entities']
    images = sorted(str(path) for path in source.iterdir() if path.is_file()
                    and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError('输入目录没有支持的图片')
    rows = reason_batch(images, knowledge, output_path=str(target), top_k=top_k)
    return {'output_file': str(target), 'n_images': len(rows),
            'n_unknown': sum(row['label'] == 'unknown' for row in rows),
            'visual': False, 'requires_human_review': True}


def review_summary(review_dir, status='', limit=20):
    from core_modules.organize.review_queue import ReviewQueue
    queue = ReviewQueue(str(_directory(review_dir)))
    return {'summary': queue.summary(), 'records': queue.list_pending(status or None, limit=limit),
            'auto_approved': True}


def prepare_review(evidence_file, review_dir):
    from core_modules.organize.review_queue import ReviewQueue
    target = _fresh_output(review_dir)
    with Path(evidence_file).expanduser().open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        row['review'] = {'status': 'pending', 'reviewer': None}
    result = ReviewQueue(str(target)).seed(rows)
    return {'review_dir': str(target), 'requires_human_review': True, **result}


def register_photo_tools(agent, organize=None, services=None):
    text = {'type': 'string', 'minLength': 1}
    limit = {'type': 'integer', 'minimum': 1, 'maximum': 100}

    def register(name, function, description, properties, required,
                 side_effects='write', contextual=False, volatile=False, aliases=None):
        agent.register_tool(name, function, description, parameters={
            'type': 'object', 'properties': properties, 'required': required,
            'additionalProperties': False,
        }, side_effects=side_effects, contextual=contextual, volatile=volatile, aliases=aliases)

    register_image_tools(agent)
    register('inspect_photos', inspect_photos,
             '只读检查图片目录、计数与文件名元数据，非视觉识别，不递归子目录。',
             {'input_dir': text, 'limit': limit}, ['input_dir'], side_effects='read')
    register('group_photos_by_actor', group_photos_by_actor,
             '按演员分类图片：根据文件名已有的「演员饰角色」标注，复制到项目输出目录下的新演员专辑。'
             '合照可进入多个专辑；未标注只计数，不猜身份。不是人脸识别。无需参数，使用当前项目目录。',
             {}, [])
    register('knowledge_summary', knowledge_summary, '读取本地剧目知识库的场景与角色。',
             {'knowledge_dir': text}, ['knowledge_dir'], side_effects='read', aliases=['知识库', '幕次', '角色'])
    register('build_play_knowledge', build_play_knowledge if services is None else
             lambda _execution_context=None, **params: services.submit('knowledge', params, _execution_context),
             '根据本地剧目信息文本构建知识库，输出必须为新目录。服务模式提交持久任务，查询完成后使用知识库路径。',
             {'info_path': text, 'play_root': text, 'output_dir': text},
             ['info_path', 'play_root', 'output_dir'], contextual=services is not None)
    scene_properties = {'input_dir': text, 'knowledge_dir': text, 'output_file': text,
                        'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 20}}
    if services is not None:
        scene_properties.update(use_embedder={'type': 'boolean'}, use_llm={'type': 'boolean'},
            visual_policy={'type': 'object', 'additionalProperties': False, 'properties': {
                'mode': {'type': 'string', 'enum': ['automatic', 'on_demand']},
                'automatic_fraction': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'max_images': {'type': 'integer', 'minimum': 1, 'maximum': 64},
                'max_output_tokens_per_request': {'type': 'integer', 'minimum': 1, 'maximum': 2048},
                'max_batch_output_tokens': {'type': 'integer', 'minimum': 1, 'maximum': 32768}}})
    register('scene_reason', scene_reason if services is None else
             lambda _execution_context=None, **params: services.submit('scene_reason', params, _execution_context),
             ('文件名与知识库闭集场景推理，非视觉识别。' if services is None else
              '提交持久场景分析；use_embedder 启用图文检索，use_llm 启用预算内视觉观察。单图明确观察用 on_demand。')
             + '结果必须人工复核，输出为新 JSONL；任务提交不代表完成。',
             scene_properties, ['input_dir', 'knowledge_dir', 'output_file'],
             contextual=services is not None, aliases=['场景推理', '场景识别', '生成候选'])
    register('review_summary', review_summary, '只读查看人工复核统计和记录。',
             {'review_dir': text, 'status': {'type': 'string', 'enum': ['', 'pending', 'approved', 'rejected']},
              'limit': limit}, ['review_dir'], side_effects='read', volatile=True, aliases=['复核队列', '审核状态'])
    register('prepare_review', prepare_review, '从场景候选 JSONL 创建新的待人工复核队列，不自动批准。',
             {'evidence_file': text, 'review_dir': text}, ['evidence_file', 'review_dir'])
    if organize is not None or services is not None:
        def safe_organize(input_dir, output_dir, db_root, **options):
            source = _directory(input_dir)
            target = _fresh_output(output_dir, source)
            database = _directory(db_root)
            return organize(input_dir=str(source), output_dir=str(target),
                            db_root=str(database), **options)

        register('organize_photos', safe_organize if services is None else
                 lambda _execution_context=None, **params: services.submit('organize', params, _execution_context),
                 '完整整理：连拍去重选优、人脸识别、时间场景分段、规范命名并复制；不删除原图。'
                 '需服务已加载视觉模型和人脸锚定库，输出必须为新目录；场景证据仅采纳人工批准标签。',
                 {'input_dir': text, 'output_dir': text, 'db_root': text,
                  'keep_per_bucket': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                  'gap_seconds': {'type': 'integer', 'minimum': 1},
                  'scene_evidence_file': text, 'identity_registry_file': text,
                  'face_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1},
                  'face_margin': {'type': 'number', 'minimum': 0, 'maximum': 2}},
                 ['input_dir', 'output_dir', 'db_root'], contextual=services is not None)
    register('export_review', export_review, '把当前人工批准的场景证据导出到新 JSONL，保留复核来源；不批准任何记录。',
             {'review_dir': text, 'output_file': text}, ['review_dir', 'output_file'])
    if services is not None:
        register('get_capabilities', services.capabilities, '查询实际服务能力与视觉模型配置，不触发模型加载。',
                 {}, [], side_effects='read', volatile=True)
        register('get_task', services.get, '查询当前项目已关联的持久任务；排队或执行中应结束本轮，完成后才可用产物。',
                 {'task_id': text}, ['task_id'], side_effects='read', volatile=True)
        register('cancel_task', services.cancel, '取消当前项目已关联的业务任务，保留检查点和原图。',
                 {'task_id': text}, ['task_id'], volatile=True)
        register('resume_task', services.resume, '恢复当前项目已关联的中断、取消或失败任务，复用检查点并校验版本。',
                 {'task_id': text}, ['task_id'], volatile=True)
        register('observe_photo', services.observe, '对指定真实图片进行预算内多模态观察，返回画面事实，不判断身份或批准幕次。',
                 {'image_path': text}, ['image_path'], side_effects='read', contextual=True)
        register('build_face_database', lambda _execution_context=None, **params:
                 services.submit('face_database', params, _execution_context),
                 '从明确标注演员的照片目录提交人脸锚定建库任务；人脸库与报告必须为新目录，不修改已有正式图库。',
                 {'organized_dirs': {'type': 'array', 'items': text, 'minItems': 1, 'maxItems': 20},
                  'db_root': text, 'output_dir': text,
                  'max_anchors': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                  'dry_run': {'type': 'boolean'}}, ['organized_dirs', 'db_root', 'output_dir'], contextual=True)
        register('scan_faces', lambda _execution_context=None, **params:
                 services.submit('face_cluster', params, _execution_context),
                 '提交演员人脸聚类候选扫描任务，不自动赋予演员身份；输出必须为新目录。',
                 {'input_dir': text, 'output_dir': text,
                  'dist_threshold': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 2},
                  'min_face_area': {'type': 'integer', 'minimum': 1, 'maximum': 100000000}},
                 ['input_dir', 'output_dir'], contextual=True)


def export_review(review_dir, output_file):
    from core_modules.organize.review_queue import ReviewQueue
    target = _fresh_output(output_file)
    if target.suffix.lower() != '.jsonl':
        raise ValueError('output_file 必须为 JSONL 文件')
    queue = ReviewQueue(str(_directory(review_dir)), snapshot_on_write=False)
    with queue.store.session() as connection:
        bindings = queue._bindings(connection)
        records = [dict(record, image=bindings.get(record['image'], record['image']))
                   for record in queue._records(connection).values()
                   if record.get('review', {}).get('status') == 'approved' and record.get('label') not in (None, 'unknown')]
    if not records:
        raise ValueError('没有人工批准的场景证据，请先完成人工复核')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return {'output_file': str(target), 'n_approved': len(records), 'evidence_type': 'human_reviewed',
            'auto_approved': True}
