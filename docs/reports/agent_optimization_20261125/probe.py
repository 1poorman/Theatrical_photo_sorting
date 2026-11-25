import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from core_modules.organize.agent_runtime import AgentRuntime
from core_modules.organize.conversational_agent import ConversationalAgent, ConversationStore, _json_text
from core_modules.organize.langchain_agent import CompatibleToolModel, build_photo_agent
from langchain_core.messages import AIMessage

module_spec = importlib.util.spec_from_file_location('agent_test_helpers', REPOSITORY / 'tests/test_langchain_agent.py')
helpers = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helpers)


def make_agent(root, responses):
    endpoint = helpers.ScriptedModel(responses=responses)
    return build_photo_agent(str(root), model=CompatibleToolModel(endpoints=[endpoint]))


findings = []

with tempfile.TemporaryDirectory(prefix='theatre-agent-audit-') as directory:
    root = Path(directory)
    legacy_calls = []
    legacy = ConversationalAgent(str(root / 'legacy'))
    legacy.register_tool('bounded', lambda limit, status: legacy_calls.append([limit, status]) or {},
                         'bounded values', parameters={
                             'type': 'object', 'properties': {
                                 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                                 'status': {'type': 'string', 'enum': ['pending', 'approved']}},
                             'required': ['limit', 'status'], 'additionalProperties': False})
    legacy.chat('/tool bounded {"limit":-10,"status":"unexpected","extra":1}')
    assert legacy_calls == [[-10, 'unexpected']]
    findings.append({'case': 'legacy_schema_constraints', 'observed': 'range and enum violations executed; unknown argument silently discarded'})

    effects = []
    runtime = AgentRuntime(str(root / 'runtime'))
    runtime.register_tool('record', lambda: effects.append('executed') or {'auto_approved': True})
    runtime.run('record', task_id='same-request')
    runtime.run('record', task_id='same-request')
    assert len(effects) == 2 and len(runtime.list_tasks()) == 1
    findings.append({'case': 'runtime_task_id_replay', 'executions': len(effects), 'stored_tasks': len(runtime.list_tasks())})

    duplicate_actions = [helpers.action('record_write', {'destination': 'new-output'}), helpers.final(),
                         helpers.action('record_write', {'destination': 'new-output'}), helpers.final()]
    duplicate = make_agent(root / 'duplicate', duplicate_actions)
    cross_turn_effects = []
    duplicate.register_tool('record_write', lambda destination: cross_turn_effects.append(destination) or {},
                            'record write', parameters={'type': 'object', 'properties': {
                                'destination': {'type': 'string'}}, 'required': ['destination']})
    duplicate.chat('write', 'same-session')
    duplicate.chat('repeat', 'same-session')
    assert len(cross_turn_effects) == 2
    findings.append({'case': 'cross_turn_deduplication', 'executions': len(cross_turn_effects)})

    result_agent = make_agent(root / 'result-memory', [
        helpers.action('publish_knowledge', {'output_dir': '/synthetic/new-kb'}), helpers.final()])
    result_agent.register_tool('publish_knowledge', lambda output_dir: {'knowledge_dir': output_dir},
                              'publish knowledge', parameters={'type': 'object', 'properties': {
                                  'output_dir': {'type': 'string'}}, 'required': ['output_dir']})
    result = result_agent.chat('build knowledge')
    assert 'knowledge_dir' not in result['memory'] and 'output_dir' in result['memory']
    findings.append({'case': 'tool_result_memory', 'knowledge_dir_saved': False, 'output_dir_saved': True})

    stale = make_agent(root / 'stale-memory', [helpers.final(), helpers.final()])
    stale.chat('project A', 'projects', {'input_dir': '/synthetic/project-a', 'output_file': '/synthetic/project-a.jsonl'})
    result = stale.chat('project B', 'projects', {'input_dir': '/synthetic/project-b'})
    assert result['memory']['output_file'] == '/synthetic/project-a.jsonl'
    findings.append({'case': 'project_memory_scope', 'old_project_output_retained': True})

    soft_failure = make_agent(root / 'soft-failure', [helpers.action('partial_organize', {}), helpers.final()])
    soft_failure.register_tool('partial_organize', lambda: {'total_input': 2, 'kept': 0, 'failed': 2},
                               'return business failure counts')
    call = soft_failure.chat('organize')['turn_tool_calls'][0]
    assert call['ok'] and call['state'] == 'review' and call['result']['failed'] == 2
    findings.append({'case': 'business_failure_status', 'call_ok': call['ok'], 'state': call['state'], 'failed_items': 2})

    false_final = make_agent(root / 'completion', [helpers.final('已完成整理并导出结果。')])
    result = false_final.chat('organize photos')
    assert not result['turn_tool_calls'] and not result['fallback']
    findings.append({'case': 'completion_evidence', 'accepted_completion_without_tool': True})

    secrets = make_agent(root / 'secrets', [helpers.final()])
    result = secrets.chat('api_key=synthetic-not-a-real-secret', context={
        'settings': {'api_key': 'synthetic-nested-value'}})
    stored = secrets.get_session(result['session_id'])
    assert 'api_key' not in result['memory'] and 'api_key' in result['memory']['settings']
    assert 'synthetic-not-a-real-secret' in stored['messages'][0]['content']
    findings.append({'case': 'redaction_boundary', 'top_level_key_filtered': True,
                     'nested_key_retained': True, 'message_retained': True})

    endpoint = helpers.ScriptedModel(responses=[AIMessage(content=json.dumps({
        'action': 'final', 'content': 'done'}), usage_metadata={
            'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17})])
    response = CompatibleToolModel(endpoints=[endpoint]).invoke('hello')
    assert response.usage_metadata is None
    findings.append({'case': 'json_protocol_usage', 'endpoint_usage_preserved': False})

    store = ConversationStore(str(root / 'summary'), prompt_history=4, summary_chars=1000)
    session = store.load('long-session')
    store.append(session, 'assistant', 'UNRESOLVED_REVIEW_TASK_OLD')
    for index in range(100):
        store.append(session, 'user', 'ordinary message ' + str(index))
    assert 'UNRESOLVED_REVIEW_TASK_OLD' not in session['summary']
    assert 'UNRESOLVED_REVIEW_TASK_OLD' not in str(store.prompt_messages(session))
    findings.append({'case': 'rolling_summary', 'old_unresolved_task_retained_in_prompt': False})

    truncated = _json_text({'rows': ['sample' * 3000], 'report_path': '/synthetic/report.json'})
    try:
        json.loads(truncated)
    except ValueError:
        findings.append({'case': 'result_truncation', 'valid_json': False,
                         'report_path_retained': 'report_path' in truncated})
    else:
        raise AssertionError('expected invalid truncated JSON')

parser = argparse.ArgumentParser(description='Offline agent boundary probes; no model endpoint or GPU calls')
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
output = arguments.output
source_files = ['core_modules/organize/agent_runtime.py',
                'core_modules/organize/conversational_agent.py',
                'core_modules/organize/langchain_agent.py',
                'core_modules/organize/photo_agent_tools.py', 'tests/test_langchain_agent.py']
source_sha256 = {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest()
                 for name in source_files}
with output.open('x', encoding='utf-8') as stream:
    json.dump({'source_sha256': source_sha256, 'cases': findings}, stream, ensure_ascii=False, indent=2)
print(json.dumps({'verified_cases': len(findings), 'report': str(output)}, ensure_ascii=False))
for finding in findings:
    print(json.dumps(finding, ensure_ascii=False))
