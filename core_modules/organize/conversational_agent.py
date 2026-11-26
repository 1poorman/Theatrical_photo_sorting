# -*- coding: utf-8 -*-
"""持久化会话智能体：连续对话、结构化记忆和受限工具自主调用。"""
import json
import os
import re
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

from core_modules.organize.agent_runtime import AgentRuntime, ToolNotAllowed
from core_modules.organize.agent_storage import SessionHistory
from core_modules.organize.agent_contracts import (
    ARTIFACT_FIELDS, COMPLETION_RE, PATH_FIELDS, SENSITIVE_KEY_RE, bounded_json,
    outcome, redact, request_key, validate_arguments,
)

try:
    import fcntl
except ImportError:  # Windows 等无 fcntl 环境仍保留进程内锁
    fcntl = None


SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
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
    return bounded_json(value, limit)


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
    """保留完整历史，聊天仅载入提示窗口并追加本轮记录。"""

    def __init__(self, root, prompt_history=24, summary_chars=6000):
        self.root = os.path.abspath(root)
        self.prompt_history = max(4, int(prompt_history))
        self.summary_chars = max(1000, int(summary_chars))

    def normalize_id(self, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError('session_id 只能包含字母、数字、点、下划线和连字符')
        return session_id

    def history(self, session_id):
        return SessionHistory(self.root, self.normalize_id(session_id))

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

    def load(self, session_id=None, for_turn=False):
        session_id = self.normalize_id(session_id)
        now = _now()
        default = dict(session_id=session_id, created_at=now, updated_at=now, memory={}, summary='')
        return self.history(session_id).read(default, self.prompt_history + 40 if for_turn else None)

    def save(self, session):
        session['updated_at'] = _now()
        return self.history(session['session_id']).checkpoint(session)

    def append(self, session, role, content, **extra):
        row = {'role': role, 'content': redact(str(content)), 'at': _now()}
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
        self.services = None

    def register_tool(self, name, fn, description, parameters=None, aliases=None,
                      side_effects='write', contextual=False, volatile=False):
        if not SESSION_ID_RE.fullmatch(name):
            raise ValueError(f'非法工具名: {name}')
        self.runtime.register_tool(name, fn, contextual=contextual)
        self.tool_specs[name] = {
            'name': name,
            'description': description,
            'parameters': parameters or {'type': 'object', 'properties': {}},
            'aliases': list(aliases or []),
            'side_effects': side_effects,
            'volatile': volatile,
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
            f'工具清单：{_json_text(self.tools(), 48000)}\n'
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

    def _merge_memory(self, session, values, source='user'):
        if not isinstance(values, dict):
            return
        values = redact(values)
        if source == 'model':
            values = {key: value for key, value in values.items() if key not in PATH_FIELDS and key != 'project_id'}
        projects = session.setdefault('projects', {})
        active = session.get('active_project_id', 'default')
        projects.setdefault(active, {'memory': session.setdefault('memory', {}), 'tasks': {}, 'artifacts': {}})
        projects[active]['memory'] = session.setdefault('memory', {})
        project_id = values.get('project_id')
        if project_id is not None and (not isinstance(project_id, str) or not SESSION_ID_RE.fullmatch(project_id)):
            raise ValueError('project_id 必须为合法的项目名称')
        if values.get('input_dir') and not isinstance(values['input_dir'], str):
            raise ValueError('input_dir 必须为路径字符串')
        if values.get('input_dir') and (project_id or source == 'user'
                or session['memory'].get('input_dir') != values['input_dir']):
            import hashlib
            namespace = project_id or session['memory'].get('project_id', '')
            digest = hashlib.sha256(str(Path(values['input_dir']).expanduser().resolve()).encode()).hexdigest()[:20]
            project_id = namespace + '.' + digest if namespace else digest
            if namespace:
                values['project_id'] = namespace
        if project_id and project_id != active:
            previous = session['memory']
            projects.setdefault(project_id, {'memory': {key: previous[key] for key in ('play',)
                                                      if key in previous and active == 'default'},
                                             'tasks': {}, 'artifacts': {}})
            session['active_project_id'] = project_id
            session['memory'] = projects[project_id]['memory']
        else:
            session.setdefault('active_project_id', active)
        memory = session.setdefault('memory', {})
        for key, value in values.items():
            if (isinstance(key, str) and len(key) <= 80
                    and not SENSITIVE_KEY_RE.search(key) and isinstance(
                    value, (str, int, float, bool, list, dict, type(None)))):
                memory[key] = value

    def _project(self, session):
        self._merge_memory(session, {})
        return session['projects'][session['active_project_id']]

    def _remember_result(self, session, tool, arguments, result):
        existing_id = outcome(result)['business_task_id']
        if tool in ('get_task', 'cancel_task', 'resume_task') and existing_id in self._project(session)['tasks']:
            previous = self._project(session)['tasks'][existing_id]
            tool, arguments = previous['tool'], previous['arguments']
        self._merge_memory(session, arguments, source='tool')
        facts = outcome(result)
        payload = result.get('result') if facts['business_task_id'] else result
        payload = payload if isinstance(payload, dict) else {}
        paths = {key: payload[key] for key in ARTIFACT_FIELDS if payload.get(key)}
        if tool == 'build_play_knowledge' and payload.get('output_dir'):
            paths['knowledge_dir'] = payload['output_dir']
        if tool == 'scene_reason' and payload.get('output_file'):
            paths['evidence_file'] = payload['output_file']
        if tool == 'export_review' and payload.get('output_file'):
            paths['scene_evidence_file'] = payload['output_file']
        self._merge_memory(session, paths, source='tool')
        project = self._project(session)
        for artifact in facts['artifacts']:
            project['artifacts'][artifact['kind']] = artifact
        if facts['business_task_id']:
            task_id = facts['business_task_id']
            record = {'tool': tool, 'arguments': arguments, 'snapshot': result, 'outcome': facts}
            if record != project['tasks'].get(task_id):
                project['tasks'][task_id] = record
                session.dirty_tasks.add((session['active_project_id'], task_id))

    def _refresh_tasks(self, session):
        if self.services is None:
            return
        for task_id, record in self._project(session)['tasks'].items():
            try:
                result = self.services.get(task_id)
            except (KeyError, ValueError, RuntimeError, OSError):
                continue
            self._remember_result(session, record['tool'], record['arguments'], result)

    def _execute_tool(self, session, spec, arguments, deadline=None):
        arguments, missing = validate_arguments(spec['parameters'], arguments)
        if missing:
            raise ValueError('调用工具前还需要参数：' + '、'.join(missing))
        if arguments.get('input_dir') and Path(arguments['input_dir']).expanduser().is_dir():
            self._merge_memory(session, {'input_dir': arguments['input_dir']}, source='tool')
        project = self._project(session)
        if spec['name'] in ('get_task', 'cancel_task', 'resume_task') and arguments['task_id'] not in project['tasks']:
            raise ValueError('只能操作当前会话项目已关联的业务任务')
        key = request_key(session['session_id'] + ':' + session['active_project_id'], spec, arguments) if (
            spec['side_effects'] == 'write' and not spec['volatile']) else None
        task = self.runtime.run(spec['name'], arguments, idempotency_key=key,
                                context={'session_id': session['session_id'], 'project_id': session['active_project_id'],
                                         'deadline': deadline})
        business_id = outcome(task.get('result'))['business_task_id']
        if task.get('reused') and business_id and self.services is not None:
            result = self.services.get(business_id)
            task = dict(task, result=result, outcome=outcome(result))
            task['state'] = {'submitted': 'waiting', 'partial': 'partial', 'failed': 'failed',
                             'interrupted': 'interrupted', 'cancelled': 'interrupted'}.get(
                                 task['outcome']['status'], 'review')
        facts = task.get('outcome') or outcome(task.get('result'))
        if task['state'] in ('failed', 'interrupted'):
            facts = dict(facts, ok=False, status=task['state'], error=task['error'])
        elif task['state'] in ('queued', 'running'):
            facts = dict(facts, ok=True, status='submitted')
        call = {'ok': facts['ok'], 'task_id': task['task_id'], 'state': task['state'],
                'result': task['result'], 'outcome': facts, 'error': task['error'],
                'reused': bool(task.get('reused')), 'project_id': session['active_project_id']}
        if isinstance(task['result'], dict):
            self._remember_result(session, spec['name'], arguments, task['result'])
        return call

    def _final_answer(self, session, answer, first_call):
        current = session['tool_calls'][first_call:]
        writes = any(self.tool_specs.get(call['tool'], {}).get('side_effects') == 'write' for call in current)
        if current and (writes or any(not call.get('ok') for call in current) or COMPLETION_RE.search(answer)):
            return '\n'.join(self._tool_result_answer(call) for call in current)
        if not current and COMPLETION_RE.search(answer):
            tasks = self._project(session)['tasks']
            if tasks:
                return '\n'.join(self._tool_result_answer(dict(record, tool=record['tool'],
                    result=record['snapshot'], ok=record['outcome']['ok'])) for record in tasks.values())
            previous = self.store.history(session['session_id']).previous_write(session['active_project_id'],
                {name for name, spec in self.tool_specs.items() if spec['side_effects'] == 'write'})
            if previous:
                return '历史已核验的执行结果：\n' + self._tool_result_answer(previous)
            return '尚无可核验的执行记录，不能确认操作已完成。请明确操作并提供所需路径。'
        return redact(answer)

    def _extract_inline_memory(self, message):
        values = {}
        for match in KEY_VALUE_RE.finditer(message or ''):
            value = match.group('value').strip('"\'')
            values[match.group('key')] = value
        return values

    def _validate_arguments(self, spec, arguments, memory):
        return validate_arguments(spec.get('parameters', {}), arguments, memory, coerce=True)

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
            return {'action': 'tool', 'tool': ranked[0][1],
                    'arguments': self._inline_path_arguments(text),
                    'reason': 'deterministic_fallback_router'}

        if any(word in text for word in ('工具', '能做什么', '帮助')):
            names = '、'.join(sorted(self.tool_specs)) or '无'
            return {'action': 'final', 'content': f'当前可用工具：{names}。'}
        return {
            'action': 'final',
            'content': '本轮内容已保存到会话记忆，但本地模型当前不可用或未返回有效指令。'
                       '请明确要执行的操作，或使用 /tool 工具名 {参数}。',
        }

    def _inline_path_arguments(self, text):
        """从自然语言消息提取路径类参数（目录/输出路径），供确定性路由使用。"""
        paths = re.findall(
            r'(?:(?:/[^\s，。；,;]+)+|~/(?:[^\s，。；,;]+)+)', text or '')
        if not paths:
            return {}
        cleaned = []
        for p in paths:
            p = p.rstrip('。，、；!？?)】》"\'')
            if p and p not in ('/', '/tmp'):
                cleaned.append(p)
        if not cleaned:
            return {}
        args = {}
        if len(cleaned) >= 1:
            args['input_dir'] = cleaned[0]
        if len(cleaned) >= 2:
            args['output_file'] = cleaned[1]
        return args

    def _tool_result_answer(self, call):
        facts = call.get('outcome') or {}
        status = facts.get('status')
        if status == 'submitted':
            return f'工具 {call.get("tool")} 已提交，尚未完成；业务任务 {facts.get("business_task_id") or call.get("task_id")}。'
        if status in ('partial', 'failed', 'cancelled', 'interrupted'):
            return f'工具 {call.get("tool")} 状态 {status}，失败条目 {facts.get("failed", 0)}：{_json_text(call.get("result"), 3000)}'
        if call.get('ok'):
            review = '，待人工复核' if facts.get('requires_human_review') else ''
            return f'工具 {call.get("tool")} 已执行{review}：{_json_text(call.get("result"), 3000)}'
        return f'工具 {call.get("tool")} 执行失败：{call.get("error")}'

    def chat(self, message, session_id=None, context=None):
        if not isinstance(message, str) or not message.strip():
            raise ValueError('message 不能为空')
        normalized_id = self.store.normalize_id(session_id)
        with self.store.turn_lock(normalized_id):
            return self._chat_locked(message, normalized_id, context)

    def _chat_locked(self, message, session_id, context=None):
        if context is not None and not isinstance(context, dict):
            raise ValueError('context 必须是 JSON 对象')
        session = self.store.load(session_id, for_turn=True)
        self._merge_memory(session, context or {})
        self._merge_memory(session, self._extract_inline_memory(message))
        self._refresh_tasks(session)
        self.store.append(session, 'user', message)
        self.store.save(session)
        first_call = len(session['tool_calls'])

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
            self._merge_memory(session, command.get('memory'), source='model')

            action = command.get('action')
            if action == 'final':
                answer = str(command.get('content') or '').strip()
                if not answer:
                    answer = '没有可展示的文本结果。'
                answer = self._final_answer(session, answer, first_call)
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
                try:
                    arguments, missing = self._validate_arguments(
                        spec, command.get('arguments') or {}, session.get('memory', {}))
                except ValueError as exc:
                    arguments, missing = {}, []
                    tool_result = {'ok': False, 'error': str(exc)}
                    spec = None
                executed_arguments = arguments
                if missing:
                    answer = '调用工具前还需要参数：' + '、'.join(missing)
                    self.store.append(session, 'assistant', answer, fallback=used_fallback)
                    path = self.store.save(session)
                    return self._response(session, answer, traces, used_fallback, path)
                try:
                    if spec is not None:
                        tool_result = self._execute_tool(session, spec, arguments)
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
            'message_count': session.older_messages + len(session['messages']),
            'tool_calls': session.get('tool_calls', []),
            'turn_tool_calls': session['tool_calls'],
            'trace': traces,
            'fallback': bool(fallback),
            'session_file': path,
            'project_id': session.get('active_project_id'),
            'workflow': self._project(session),
            'execution_report': [dict(tool=record['tool'], **record['outcome'])
                                 for record in self._project(session)['tasks'].values()],
            'operation_report': [dict(tool=call['tool'], task_id=call.get('task_id'), **call['outcome'])
                                 for call in session.get('tool_calls', [])
                                 if call.get('project_id') == session.get('active_project_id') and call.get('outcome')],
        }
