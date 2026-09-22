"""独立于 Web/GPU 服务的剧照工具；重型模块按需加载。"""
import json
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


def knowledge_summary(knowledge_dir):
    root = _directory(knowledge_dir)
    with (root / 'entities.json').open(encoding='utf-8') as stream:
        data = json.load(stream)
    return {'knowledge_dir': str(root), 'scenes': data.get('scenes', []),
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
    knowledge = knowledge_summary(knowledge_dir)
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


def register_photo_tools(agent, organize=None):
    text = {'type': 'string', 'minLength': 1}
    limit = {'type': 'integer', 'minimum': 1, 'maximum': 100}

    def register(name, function, description, properties, required):
        agent.register_tool(name, function, description, parameters={
            'type': 'object', 'properties': properties, 'required': required,
            'additionalProperties': False,
        })

    register_image_tools(agent)
    register('inspect_photos', inspect_photos,
             '只读检查图片目录、计数与文件名元数据，非视觉识别，不递归子目录。',
             {'input_dir': text, 'limit': limit}, ['input_dir'])
    register('knowledge_summary', knowledge_summary, '读取本地剧目知识库的场景与角色。',
             {'knowledge_dir': text}, ['knowledge_dir'])
    register('build_play_knowledge', build_play_knowledge,
             '根据本地剧目信息文本和剧目目录构建知识库，不抓取网络，输出必须为新目录。',
             {'info_path': text, 'play_root': text, 'output_dir': text},
             ['info_path', 'play_root', 'output_dir'])
    register('scene_reason', scene_reason,
             '使用文件名与知识库证据进行闭集场景推理；不做视觉识别，结果必须人工复核。输出为新 JSONL。',
             {'input_dir': text, 'knowledge_dir': text, 'output_file': text,
              'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 20}},
             ['input_dir', 'knowledge_dir', 'output_file'])
    register('review_summary', review_summary, '只读查看人工复核统计和记录。',
             {'review_dir': text, 'status': {'type': 'string', 'enum': ['', 'pending', 'approved', 'rejected']},
              'limit': limit}, ['review_dir'])
    register('prepare_review', prepare_review, '从场景候选 JSONL 创建新的待人工复核队列，不自动批准。',
             {'evidence_file': text, 'review_dir': text}, ['evidence_file', 'review_dir'])
    if organize is not None:
        def safe_organize(input_dir, output_dir, db_root, **options):
            source = _directory(input_dir)
            target = _fresh_output(output_dir, source)
            database = _directory(db_root)
            return organize(input_dir=str(source), output_dir=str(target),
                            db_root=str(database), **options)

        register('organize_photos', safe_organize,
                 '完整整理：连拍去重选优、人脸识别、时间场景分段、规范命名并复制；不删除原图。'
                 '需服务已加载视觉模型和人脸锚定库，输出必须为新目录；场景证据仅采纳人工批准标签。',
                 {'input_dir': text, 'output_dir': text, 'db_root': text,
                  'keep_per_bucket': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                  'gap_seconds': {'type': 'integer', 'minimum': 1},
                  'scene_evidence_file': text}, ['input_dir', 'output_dir', 'db_root'])
