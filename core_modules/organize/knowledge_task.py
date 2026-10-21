"""Per-media knowledge checkpoints; publishing never repeats media processing."""
from pathlib import Path

from core_modules.organize.play_knowledge import build_knowledge, _media_record, parse_plot_text, read_knowledge
from core_modules.organize.review_store import atomic_rows
from core_modules.organize.task_artifacts import atomic_json, fingerprint
from core_modules.organize.task_versions import execution_code


class KnowledgeTask:
    def __init__(self, params):
        self.params = params
        self.context = None
        self.scenes = parse_plot_text(params['sources']['text'])[1]
        self.code_version = execution_code([
            'core_modules/organize/knowledge_task.py', 'core_modules/organize/play_knowledge.py',
            'core_modules/organize/play_source.py', 'core_modules/organize/filename_parser.py',
            'core_modules/organize/task_artifacts.py'])

    def model_contract(self):
        return {'code': self.code_version, 'model': None}

    def process(self, item):
        image = item['image']
        if fingerprint(image) != item['sha256']:
            raise ValueError('source_content_changed')
        row = _media_record(image, self.params['play_root'], self.scenes)
        if fingerprint(image) != item['sha256']:
            raise ValueError('source_content_changed_during_ingest')
        return {'image': image, 'stage': 'media', 'media': row}

    def advance(self):
        return False

    def finish(self, rows):
        params = self.params
        if fingerprint(params['info_path']) != params['info_sha256']:
            raise ValueError('plot_content_changed')
        for row in rows:
            self.context.check()
            if not row.get('error') and fingerprint(row['image']) != params['input_hashes'][row['image']]:
                raise ValueError('completed_source_content_changed')
        knowledge = build_knowledge(params['info_path'], params['play_root'], sources=params['sources'],
                                    media_records=[row['media'] for row in rows if not row.get('error')],
                                    filename_scenes=params['filename_scenes'])
        output = Path(params['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        if not (output / 'knowledge_snapshot.json').exists() and (output / 'entities.json').exists() and not (output / '.knowledge_pending.json').exists():
            atomic_json(output / 'knowledge_snapshot.json', read_knowledge(output))
        # A first publication must not expose partial legacy files before commit.
        atomic_json(output / '.knowledge_pending.json', {'publishing': True})
        publications = [
            ('documents.jsonl', knowledge['documents'], atomic_rows),
            ('media.jsonl', knowledge['media'], atomic_rows),
            ('entities.json', {key: knowledge[key] for key in ('scenes', 'roles', 'role_aliases')}, atomic_json),
            ('knowledge_report.json', {'parser': knowledge['parser'], 'urls': knowledge['urls'], 'stats': knowledge['stats'],
                                      'execution_contract': self.context.checkpoint('execution_contract'),
                                      'failed': sum(bool(row.get('error')) for row in rows),
                                      'errors': [row for row in rows if row.get('error')]}, atomic_json)]
        for name, value, publisher in publications:
            self.context.check()
            publisher(output / name, value)
        self.context.check()
        # Readers use this one atomic generation; partial compatibility exports
        # after a crash cannot mix new entities with an older report.
        atomic_json(output / 'knowledge_snapshot.json', {
            'entities': publications[2][1], 'report': publications[3][1]})
        return {'output_dir': str(output), 'stats': knowledge['stats'],
                'scenes': [scene['label'] for scene in knowledge['scenes']], 'n_roles': len(knowledge['roles']),
                'failed': sum(bool(row.get('error')) for row in rows)}
