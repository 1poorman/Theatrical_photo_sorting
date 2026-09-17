# -*- coding: utf-8 -*-
"""持久化会话智能体：连续对话、结构化记忆和受限工具自主调用。"""
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from core_modules.organize.agent_runtime import AgentRuntime, ToolNotAllowed

try:
    import fcntl
except ImportError:  # Windows 等无 fcntl 环境仍保留进程内锁
    fcntl = None


SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
SENSITIVE_KEY_RE = re.compile(
    r'(?:^|_)(?:api_?key|key|token|password|passwd|secret|authorization)(?:$|_)',
    re.I,
)
KEY_VALUE_RE = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*'
    r'(?P<value>"[^"]*"|\'[^\']*\'|[^\s,，;；]+)')
_STORE_LOCK = threading.RLock()
_SESSION_LOCKS = {}


def _session_lock(session_id):
    with _STORE_LOCK:
        return _SESSION_LOCKS.setdefault(session_id, threading.RLock())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_text(value, limit=12000):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[:limit] + '...<truncated>'
    return text


def _json_object(text):
    if not text:
        return None
    value = str(text).strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', value, re.S)
    if fence:
        value = fence.group(1).strip()
    start, end = value.find('{'), value.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start:end + 1])
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class ConversationStore:
    """将完整消息历史、滚动摘要和结构化记忆原子写入 JSON。"""

    def __init__(self, root, prompt_history=24, summary_chars=6000):
        self.root = os.path.abspath(root)
        self.prompt_history = max(4, int(prompt_history))
        self.summary_chars = max(1000, int(summary_chars))

    def normalize_id(self, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError('session_id 只能包含字母、数字、点、下划线和连字符')
        return session_id

    def _path(self, session_id):
        session_id = self.normalize_id(session_id)
        return os.path.join(self.root, session_id + '.json')

    @contextmanager
    def turn_lock(self, session_id):
        """同一会话整轮串行；Linux 下同时保护多 worker 进程。"""
        session_id = self.normalize_id(session_id)
        lock = _session_lock(session_id)
        with lock:
            os.makedirs(self.root, exist_ok=True)
            lock_path = os.path.join(self.root, session_id + '.lock')
            with open(lock_path, 'a+', encoding='utf-8') as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self, session_id=None):
        session_id = self.normalize_id(session_id)
        path = self._path(session_id)
        with _STORE_LOCK:
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    return json.load(f)
        now = _now()
        return {
            'session_id': session_id,
            'created_at': now,
            'updated_at': now,
            'memory': {},
            'summary': '',
            'messages': [],
            'tool_calls': [],
        }

    def save(self, session):
        session_id = self.normalize_id(session.get('session_id'))
        session['session_id'] = session_id
        session['updated_at'] = _now()
        path = self._path(session_id)
        os.makedirs(self.root, exist_ok=True)
        tmp = path + '.' + uuid.uuid4().hex[:8] + '.tmp'
        with _STORE_LOCK:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(session, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, path)
        return path

    def append(self, session, role, content, **extra):
        row = {'role': role, 'content': str(content), 'at': _now()}
        row.update(extra)
        session.setdefault('messages', []).append(row)
        self._refresh_summary(session)
        return row

    def _refresh_summary(self, session):
        messages = session.get('messages', [])
        if len(messages) <= self.prompt_history:
            return
        older = messages[:-self.prompt_history]
        lines = []
        for row in older[-40:]:
            content = row.get('content', '').replace('\n', ' ').strip()
            lines.append(f"{row.get('role', 'unknown')}: {content[:240]}")
        session['summary'] = '\n'.join(lines)[-self.summary_chars:]

    def prompt_messages(self, session):
        rows = session.get('messages', [])[-self.prompt_history:]
        out = []
        for row in rows:
            role = row.get('role')
            if role == 'tool':
                out.append({'role': 'user', 'content': '[工具结果] ' + row.get('content', '')})
            elif role in ('user', 'assistant'):
                out.append({'role': role, 'content': row.get('content', '')})
        return out


class ConversationalAgent:
    """通过 JSON 行动协议驱动白名单工具，并在模型失败时确定性降级。"""

    def __init__(self, root, client=None, runtime=None, max_steps=4,
                 prompt_history=24):
        self.root = os.path.abspath(root)
        self.client = client
        self.max_steps = max(1, min(8, int(max_steps)))
        self.store = ConversationStore(
            os.path.join(self.root, 'conversations'),
            prompt_history=prompt_history,
        )
        self.runtime = runtime or AgentRuntime(os.path.join(self.root, 'tasks'))
        self.tool_specs = {}

    def register_tool(self, name, fn, description, parameters=None, aliases=None):
        if not SESSION_ID_RE.fullmatch(name):
            raise ValueError(f'非法工具名: {name}')
        self.runtime.register_tool(name, fn)
        self.tool_specs[name] = {
            'name': name,
            'description': description,
            'parameters': parameters or {'type': 'object', 'properties': {}},
            'aliases': list(aliases or []),
        }
        return name

    def tools(self):
        return [self.tool_specs[k] for k in sorted(self.tool_specs)]

    def get_session(self, session_id):
        return self.store.load(session_id)

    def _system_prompt(self, session):
        return (
            '你是剧照整理系统中的受限智能体。你可以连续对话，并可自主调用白名单工具。\n'
            '只输出一个 JSON 对象，不要输出 Markdown。允许两种格式：\n'
            '{"action":"tool","tool":"工具名","arguments":{},"reason":"原因",'
            '"memory":{}}\n'
            '{"action":"final","content":"给用户的回答","memory":{}}\n'
            '规则：工具名必须来自工具清单；参数不足时不要猜测，直接 final 询问；'
            '工具失败后可换工具、修正参数或解释失败；不得请求 shell、删除或越权网络访问。\n'
            f'工具清单：{_json_text(self.tools())}\n'
            f'长期记忆：{_json_text(session.get("memory", {}))}\n'
            f'较早对话摘要：{session.get("summary") or "无"}'
        )

    def _model_command(self, session):
        if self.client is None:
            return None, {'fallback_reason': 'model_unavailable'}
        messages = [{'role': 'system', 'content': self._system_prompt(session)}]
        messages.extend(self.store.prompt_messages(session))
        try:
            result = self.client.chat(messages)
        except Exception as exc:
            return None, {'fallback_reason': f'{type(exc).__name__}: {exc}'}
        command = _json_object(result.get('content'))
        trace = dict(result.get('trace') or {})
        if command is not None:
            return command, trace

        # 端点可用但小模型没有遵循行动协议时，显式尝试大模型兜底。
        big = getattr(self.client, 'big', None)
        if big is not None and getattr(self.client, 'cascade_enabled', True):
            try:
                retried = big.chat(messages)
                command = _json_object(retried.get('content'))
                retry_trace = dict(retried.get('trace') or {})
                retry_trace['cascade_upgraded_from'] = getattr(
                    getattr(self.client, 'small', None), 'model', 'small')
                retry_trace['cascade_reason'] = 'invalid_agent_json'
                if command is not None:
                    return command, retry_trace
                trace.update(retry_trace)
            except Exception as exc:
                trace['cascade_error'] = f'{type(exc).__name__}: {exc}'
        trace['fallback_reason'] = 'invalid_agent_json'
        return None, trace

    def _merge_memory(self, session, values):
        if not isinstance(values, dict):
            return
        memory = session.setdefault('memory', {})
        for key, value in values.items():
            if (isinstance(key, str) and len(key) <= 80
                    and not SENSITIVE_KEY_RE.search(key) and isinstance(
                    value, (str, int, float, bool, list, dict, type(None)))):
                memory[key] = value

    def _extract_inline_memory(self, message):
        values = {}
        for match in KEY_VALUE_RE.finditer(message or ''):
            value = match.group('value').strip('"\'')
            values[match.group('key')] = value
        return values

    def _validate_arguments(self, spec, arguments, memory):
        if not isinstance(arguments, dict):
            arguments = {}
        properties = spec.get('parameters', {}).get('properties', {})
        allowed = set(properties)
        merged = {k: v for k, v in memory.items() if k in allowed}
        merged.update({k: v for k, v in arguments.items() if k in allowed})
        for key, value in list(merged.items()):
            expected = (properties.get(key) or {}).get('type')
            if expected == 'integer' and isinstance(value, str):
                try:
                    merged[key] = int(value)
                except ValueError:
                    pass
            elif expected == 'number' and isinstance(value, str):
                try:
                    merged[key] = float(value)
                except ValueError:
                    pass
            elif expected == 'boolean' and isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ('1', 'true', 'yes', 'on'):
                    merged[key] = True
                elif lowered in ('0', 'false', 'no', 'off'):
                    merged[key] = False
        missing = [k for k in spec.get('parameters', {}).get('required', [])
                   if merged.get(k) in (None, '')]
        return merged, missing

    def _fallback_command(self, message, session):
        text = (message or '').strip()
        explicit = re.match(r'^/tool\s+([A-Za-z0-9_.-]+)(?:\s+(\{.*\}))?$', text, re.S)
        if explicit:
            args = _json_object(explicit.group(2) or '{}') or {}
            return {'action': 'tool', 'tool': explicit.group(1), 'arguments': args,
                    'reason': 'explicit_tool_command'}

        lowered = text.lower()
        ranked = []
        for name, spec in self.tool_specs.items():
            aliases = [name] + spec.get('aliases', [])
            hits = sum(1 for alias in aliases if alias and alias.lower() in lowered)
            if hits:
                ranked.append((hits, name))
        if ranked:
            ranked.sort(reverse=True)
            return {'action': 'tool', 'tool': ranked[0][1], 'arguments': {},
                    'reason': 'deterministic_fallback_router'}

        if any(word in text for word in ('工具', '能做什么', '帮助')):
            names = '、'.join(sorted(self.tool_specs)) or '无'
            return {'action': 'final', 'content': f'当前可用工具：{names}。'}
        return {
            'action': 'final',
            'content': '本轮内容已保存到会话记忆，但本地模型当前不可用或未返回有效指令。'
                       '请明确要执行的操作，或使用 /tool 工具名 {参数}。',
        }

    def _tool_result_answer(self, call):
        if call.get('ok'):
            return f'工具 {call.get("tool")} 已执行：{_json_text(call.get("result"), 3000)}'
        return f'工具 {call.get("tool")} 执行失败：{call.get("error")}'

    def chat(self, message, session_id=None, context=None):
        if not isinstance(message, str) or not message.strip():
            raise ValueError('message 不能为空')
        normalized_id = self.store.normalize_id(session_id)
        with self.store.turn_lock(normalized_id):
            return self._chat_locked(message, normalized_id, context)

    def _chat_locked(self, message, session_id, context=None):
        session = self.store.load(session_id)
        self._merge_memory(session, context or {})
        self._merge_memory(session, self._extract_inline_memory(message))
        self.store.append(session, 'user', message)

        traces = []
        used_fallback = False
        for step in range(self.max_steps):
            command, trace = self._model_command(session)
            traces.append(trace)
            if command is None:
                # 工具已经执行后模型失效时直接用现有结果收尾，避免重复副作用。
                if (session.get('messages')
                        and session['messages'][-1].get('role') == 'tool'
                        and session.get('tool_calls')):
                    used_fallback = True
                    answer = self._tool_result_answer(session['tool_calls'][-1])
                    self.store.append(session, 'assistant', answer,
                                      trace=trace, fallback=True)
                    path = self.store.save(session)
                    return self._response(session, answer, traces, True, path)
                command = self._fallback_command(message, session)
                used_fallback = True
            self._merge_memory(session, command.get('memory'))

            action = command.get('action')
            if action == 'final':
                answer = str(command.get('content') or '').strip()
                if not answer:
                    answer = '任务已处理，但没有可展示的文本结果。'
                self.store.append(session, 'assistant', answer,
                                  trace=trace, fallback=used_fallback)
                path = self.store.save(session)
                return self._response(session, answer, traces, used_fallback, path)

            if action != 'tool':
                command = self._fallback_command(message, session)
                used_fallback = True
                if command.get('action') == 'final':
                    answer = command['content']
                    self.store.append(session, 'assistant', answer, fallback=True)
                    path = self.store.save(session)
                    return self._response(session, answer, traces, True, path)

            tool_name = command.get('tool')
            spec = self.tool_specs.get(tool_name)
            executed_arguments = command.get('arguments') or {}
            if spec is None:
                tool_result = {'ok': False, 'error': f'工具不在白名单内: {tool_name}'}
            else:
                arguments, missing = self._validate_arguments(
                    spec, command.get('arguments'), session.get('memory', {}))
                executed_arguments = arguments
                if missing:
                    answer = '调用工具前还需要参数：' + '、'.join(missing)
                    self.store.append(session, 'assistant', answer, fallback=used_fallback)
                    path = self.store.save(session)
                    return self._response(session, answer, traces, used_fallback, path)
                try:
                    task = self.runtime.run(tool_name, arguments)
                    tool_result = {
                        'ok': task.get('state') != 'failed',
                        'task_id': task.get('task_id'),
                        'state': task.get('state'),
                        'result': task.get('result'),
                        'error': task.get('error'),
                    }
                    if tool_result['ok']:
                        self._merge_memory(session, arguments)
                except (ToolNotAllowed, TypeError, ValueError) as exc:
                    tool_result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}

            call = {'at': _now(), 'step': step + 1, 'tool': tool_name,
                    'arguments': executed_arguments, **tool_result}
            session.setdefault('tool_calls', []).append(call)
            self.store.append(session, 'tool', _json_text(call), tool=tool_name)
            self.store.save(session)

            # 没有模型时，确定性路由执行一次工具后直接给出结果摘要。
            if self.client is None or used_fallback:
                answer = self._tool_result_answer(call)
                self.store.append(session, 'assistant', answer, fallback=True)
                path = self.store.save(session)
                return self._response(session, answer, traces, True, path)

        answer = '已达到本轮最大工具调用步数，任务已停止并保留全部执行记录，请人工确认后继续。'
        self.store.append(session, 'assistant', answer, fallback=True)
        path = self.store.save(session)
        return self._response(session, answer, traces, True, path)

    def _response(self, session, answer, traces, fallback, path):
        return {
            'session_id': session['session_id'],
            'answer': answer,
            'memory': session.get('memory', {}),
            'message_count': len(session.get('messages', [])),
            'tool_calls': session.get('tool_calls', []),
            'trace': traces,
            'fallback': bool(fallback),
            'session_file': path,
        }
