"""Task-level live agent smoke evaluation using isolated outputs and env models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='Evaluate real agent routing, artifacts and optional vision')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['auto', 'small', 'big'], default='big')
    parser.add_argument('--protocol', choices=['json', 'native'], default='json')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--env-path', default=str(ROOT / '.env'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output 必须为新目录')
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    os.environ['TRANS_TASK_DIR'] = str(output / 'tasks')

    from PIL import Image
    from app.agent_services import PhotoTaskService
    from app.organize_tasks import manager, shutdown
    from core_modules.organize.langchain_agent import build_photo_agent

    photos = output / 'photos'
    photos.mkdir()
    Image.new('RGB', (64, 48), 'blue').save(photos / 'neutral.png')
    info = output / 'plot.txt'
    info.write_text('话剧《测试剧目》\n第1幕：人物来到舞台。\n第2幕：人物离开舞台。', encoding='utf-8')
    service = PhotoTaskService(output / 'agent', env_path=args.env_path)
    agent = build_photo_agent(str(output / 'agent'), selection=args.model, protocol=args.protocol,
                             env_path=args.env_path, services=service, output_token_budget=16384)
    cases = []

    def run(name, message, session, context, check):
        started = time.monotonic()
        result = agent.chat(message, session, context)
        verified, detail = check(result)
        cases.append({'name': name, 'passed': bool(verified), 'detail': detail,
                      'elapsed_seconds': round(time.monotonic() - started, 3),
                      'response': result})
        print(json.dumps({'case': name, 'passed': bool(verified), 'elapsed_seconds': cases[-1]['elapsed_seconds']}, ensure_ascii=False), flush=True)
        if result['fallback'] and not result['turn_tool_calls']:
            raise RuntimeError('模型未能执行请求：' + str(result['trace']))
        return result

    try:
        run('inspect', '请检查当前照片目录，报告图片数量。', 'inspection', {'input_dir': str(photos)},
            lambda result: (any(call['tool'] == 'inspect_photos' and call.get('ok') and
                                call['result']['n_images'] == 1 for call in result['turn_tool_calls']), 'real directory count'))
        tags = output / 'tags.jsonl'
        run('manifest', '请生成图片清单，场景保持 unknown 并待人工审核，使用提供的输出路径。', 'manifest',
            {'input_dir': str(photos), 'output_file': str(tags)},
            lambda result: (tags.exists() and len(tags.read_text().splitlines()) == 1
                and all(row['scene'] == 'unknown' and row['review']['status'] == 'pending'
                for row in (json.loads(line) for line in tags.read_text().splitlines())), 'manifest content and pending status'))
        before = hashlib.sha256(tags.read_bytes()).hexdigest() if tags.exists() else None
        run('repeat_write', '再次处理同一清单请求，复用已有结果，禁止覆盖原文件。', 'manifest', {},
            lambda result: (before is not None and tags.exists() and hashlib.sha256(tags.read_bytes()).hexdigest() == before,
                            'existing output bytes preserved'))
        run('missing_arguments', '请整理照片，但我还没提供输入目录、输出目录或人脸库路径。请说明需要补充什么。',
            'missing', {}, lambda result: (not any(call.get('ok') and call['tool'] == 'organize_photos'
                                                  for call in result['turn_tool_calls']), 'no organize side effect'))
        knowledge = output / 'knowledge'
        submitted = run('knowledge_submit', '请根据剧情文件和剧目目录构建知识库，提交任务后告诉我任务编号。', 'knowledge',
            {'input_dir': str(photos), 'info_path': str(info), 'play_root': str(photos), 'output_dir': str(knowledge)},
            lambda result: (any(call['tool'] == 'build_play_knowledge' and call.get('outcome', {}).get('business_task_id')
                                for call in result['turn_tool_calls']), 'persistent knowledge task submitted'))
        task_ids = [call['outcome']['business_task_id'] for call in submitted['turn_tool_calls']
                    if call.get('outcome', {}).get('business_task_id')]
        if task_ids:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and manager().get(task_ids[-1])['status'] in ('queued', 'running'):
                time.sleep(.05)
            run('knowledge_resume', '查询刚才的知识库构建结果，告诉我实际产物路径。', 'knowledge', {},
                lambda result: (result['memory'].get('knowledge_dir') == str(knowledge)
                    and (knowledge / 'knowledge_snapshot.json').exists(), 'artifact mapping persisted across turns'))
        if args.image:
            image = args.image.expanduser().resolve()
            run('visual_observation', '请使用 observe_photo 实际观察指定图片，只描述画面可见事实，不推测演员身份或幕次。',
                'visual', {'image_path': str(image)}, lambda result: (any(
                    call['tool'] == 'observe_photo' and call.get('ok') and call['result'].get('evidence_type') == 'visual_observation'
                    and call['result']['trace'].get('called') for call in result['turn_tool_calls']), 'real image sent to configured vision endpoint'))
    finally:
        shutdown()
        source_files = ['core_modules/organize/agent_contracts.py', 'core_modules/organize/agent_budget.py',
                        'core_modules/organize/agent_runtime.py', 'core_modules/organize/conversational_agent.py',
                        'core_modules/organize/langchain_agent.py', 'core_modules/organize/photo_agent_tools.py',
                        'core_modules/organize/agent_tools.py', 'core_modules/organize/task_queue.py',
                        'app/agent_services.py', 'app/organize_tasks.py', 'app/main.py',
                        'tools/validation/evaluate_photo_agent.py']
        report = {'model_selection': args.model, 'protocol': args.protocol,
                  'scope': 'live task smoke; no scene or identity quality ground truth',
                  'expected_cases': 7 if args.image else 6,
                  'execution_complete': len(cases) == (7 if args.image else 6),
                  'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
                  'cases': cases, 'passed': sum(case['passed'] for case in cases), 'total': len(cases)}
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'total': report['total'], 'report': str(output / 'report.json')}, ensure_ascii=False))
    if not cases or not all(case['passed'] for case in cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
