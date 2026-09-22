"""独立对话入口：python -m app.photo_agent_cli，不加载视觉模型。"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='LangChain 戏剧照片整理智能体')
    parser.add_argument('--root', default='outputs/agent')
    parser.add_argument('--session', default=None)
    parser.add_argument('--model', choices=['auto', 'small', 'big'], default='auto')
    parser.add_argument('--protocol', choices=['json', 'native'], default='json')
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--context', default='{}')
    parser.add_argument('--message', help='单轮执行并输出 JSON；省略则进入连续对话')
    args = parser.parse_args()
    from core_modules.organize.langchain_agent import build_photo_agent
    agent = build_photo_agent(args.root, selection=args.model, protocol=args.protocol,
                              max_steps=args.max_steps)
    context = json.loads(args.context)
    if not isinstance(context, dict):
        parser.error('--context 必须是 JSON 对象')
    session_id = args.session
    if args.message:
        print(json.dumps(agent.chat(args.message, session_id, context), ensure_ascii=False, indent=2))
        return
    print('戏剧照片整理智能体（输入 /exit 退出）')
    while True:
        try:
            message = input('你> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message == '/exit':
            break
        if not message:
            continue
        result = agent.chat(message, session_id, context)
        session_id = result['session_id']
        context = {}
        print(f"智能体 [{session_id}]> {result['answer']}")


if __name__ == '__main__':
    main()
