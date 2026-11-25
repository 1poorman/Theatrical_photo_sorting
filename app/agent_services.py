"""Agent access to existing persistent tasks and budgeted visual evidence."""
import json
from pathlib import Path

from core_modules.organize.llm_client import DEFAULT_ENV_PATH


class PhotoTaskService:
    def __init__(self, root, capabilities=None, env_path=DEFAULT_ENV_PATH,
                 submitter=None, task_reader=None, task_cancel=None, task_resume=None,
                 vision_client=None):
        self.root = Path(root)
        self.capability_reader = capabilities or (lambda: {})
        self.env_path = env_path
        self.submitter = submitter
        self.task_reader = task_reader
        self.task_cancel = task_cancel
        self.task_resume = task_resume
        self.vision_client = vision_client

    def capabilities(self):
        from core_modules.organize.llm_client import vision_client_from_env
        client = self.vision_client or vision_client_from_env(self.env_path)
        return {'persistent_tasks': True, 'visual_observation': client is not None,
                'vision_model': getattr(client, 'model', None),
                'human_review_required': True, **self.capability_reader()}

    def submit(self, kind, params, _execution_context=None):
        from app.organize_tasks import submit_business_task
        params = dict(params)
        for field in ('input_dir', 'output_dir', 'output_file', 'play_root', 'info_path',
                      'knowledge_dir', 'db_root', 'scene_evidence_file', 'identity_registry_file'):
            if params.get(field):
                params[field] = str(Path(params[field]).expanduser().resolve())
        if params.get('organized_dirs'):
            params['organized_dirs'] = [str(Path(value).expanduser().resolve()) for value in params['organized_dirs']]
        sources = [Path(params[field]) for field in ('input_dir', 'play_root') if params.get(field)]
        sources.extend(Path(value) for value in params.get('organized_dirs', []))
        outputs = [Path(params[field]) for field in ('output_dir', 'output_file') if params.get(field)]
        if kind == 'face_database':
            outputs.append(Path(params['db_root']))
        if any(target == source or source in target.parents or target in source.parents
               for target in outputs for source in sources):
            raise ValueError('输出必须位于输入目录之外')
        capabilities = self.capabilities()
        if kind in ('organize', 'face_database', 'face_cluster') and not capabilities.get('face_recognition'):
            raise ValueError('当前服务未初始化人脸模型')
        if kind == 'scene_reason' and params.get('use_llm') and not capabilities['visual_observation']:
            raise ValueError('当前服务未配置视觉模型，不会静默退回文件名分析')
        if kind == 'scene_reason' and params.get('use_embedder') and not capabilities.get('visual_retrieval'):
            raise ValueError('当前服务不提供图文检索')
        if kind == 'organize':
            params['classify_role'] = False
        submitter = self.submitter or submit_business_task
        key = (_execution_context or {}).get('idempotency_key')
        if not key:
            raise ValueError('持久任务提交必须具有幂等键')
        return submitter(kind, params, key, fresh_output=True)

    def get(self, task_id):
        if self.task_reader:
            return self.task_reader(task_id)
        from app.organize_tasks import manager
        return manager().get(task_id)

    def cancel(self, task_id):
        if self.task_cancel:
            return self.task_cancel(task_id)
        from app.organize_tasks import manager
        return manager().cancel(task_id)

    def resume(self, task_id):
        if self.task_resume:
            return self.task_resume(task_id)
        from app.organize_tasks import manager
        return manager().resume(task_id)

    def observe(self, image_path, _execution_context=None):
        from core_modules.organize.llm_client import vision_client_from_env
        from core_modules.organize.scene_reasoner import SceneReasoner
        from core_modules.organize.task_artifacts import fingerprint
        from core_modules.organize.visual_evidence import VisualEvidenceController, VisualPolicy
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise ValueError('image_path 必须为现有图片')
        client = self.vision_client or vision_client_from_env(self.env_path)
        if client is None:
            raise ValueError('当前服务未配置视觉模型')
        deadline = (_execution_context or {}).get('deadline')
        if deadline is not None:
            import copy
            import time
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('本轮时间预算已耗尽')
            client = copy.copy(client)
            client.timeout = min(getattr(client, 'timeout', 60), remaining)
        digest = fingerprint(image)
        self.root.mkdir(parents=True, exist_ok=True)
        controller = VisualEvidenceController(client, str(self.root / 'visual_cache'),
            str(self.root / 'visual_budget.sqlite3'), digest, 1,
            VisualPolicy(mode='on_demand', max_images=1, max_batch_output_tokens=2048))
        reasoner = SceneReasoner({'scenes': []}, vision_client=client)
        evidence, trace = controller.observe(str(image), ['explicit_user_observation'], reasoner._observe)
        if not evidence:
            return {'ok': False, 'error': trace.get('error') or trace.get('skipped'),
                    'trace': trace, 'vision_budget': controller.budget.summary()}
        return {'image_path': str(image), 'source_image_sha256': digest,
                'observation': json.loads(evidence['quote']), 'evidence': evidence,
                'evidence_type': 'visual_observation', 'requires_human_review': True,
                'trace': trace, 'vision_budget': controller.budget.summary()}
