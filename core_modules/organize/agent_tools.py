import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def scan_dir_and_tag(input_dir, output_file):
    source = Path(input_dir).expanduser()
    if not source.is_dir():
        raise ValueError('input_dir 必须是目录')
    target = Path(output_file).expanduser().absolute()
    if target.suffix.lower() != '.jsonl':
        raise ValueError('output_file 必须是 JSONL 文件')
    if target.exists() or target.is_symlink():
        raise FileExistsError('输出文件已存在，请指定新路径')
    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and not path.is_symlink()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError('输入目录没有支持的图片文件')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        for image in images:
            stream.write(json.dumps({
                'image': str(image),
                'scene': 'unknown',
                'tag_source': 'deterministic_placeholder',
                'review': {'status': 'pending'},
            }, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return {'output_file': str(target), 'n_images': len(images),
            'tag_source': 'deterministic_placeholder'}


def register_image_tools(agent):
    agent.register_tool(
        'scan_dir_and_tag', scan_dir_and_tag,
        '扫描指定目录的图片，写入指定 JSONL；场景仅标为 unknown，非视觉识别；不覆盖已有文件。',
        parameters={
            'type': 'object',
            'properties': {
                'input_dir': {'type': 'string'},
                'output_file': {'type': 'string'},
            },
            'required': ['input_dir', 'output_file'],
        },
        aliases=['目录标注', '图片标注', '标注', '打标签'],
    )
