"""Recoverable organize stages using the existing quality and identity rules."""
import os
from pathlib import Path

import cv2

from core_modules.organize.burst_dedup import BurstDeduplicator, image_quality
from core_modules.organize.scene_split import split_scenes
from core_modules.organize.smart_organizer import apply_scene_evidence
from core_modules.organize.task_artifacts import atomic_json, copy_verified, fingerprint, gallery_hashes
from core_modules.organize.task_versions import model_identity, execution_code


class OrganizeTask:
    def __init__(self, params, organizer):
        self.params = params
        self.organizer = organizer
        self.context = None
        self.code_version = execution_code([
            'app/organize_tasks.py',
            'core_modules/organize/organize_task.py', 'core_modules/organize/smart_organizer.py',
            'core_modules/organize/identity_registry.py',
            'core_modules/organize/burst_dedup.py', 'core_modules/organize/scene_split.py',
            'core_modules/organize/scene_change_points.py',
            'core_modules/organize/filename_parser.py', 'core_modules/organize/role_classifier.py',
            'core_modules/organize/face_db_builder.py', 'core_modules/shot_classify.py',
            'core_modules/face_recognition/arcface/arcface_onnx.py',
            'core_modules/face_recognition/scrfd/scrfd_det.py',
            'core_modules/image_search.py', 'core_modules/organize/embedding_gemma.py',
            'core_modules/organize/task_artifacts.py', 'core_modules/organize/feature_cache.py'])

    def model_contract(self):
        from config.base import KNOWN_FACE_THRESHOLD, FACE_IMAGE_MAX_SIDE, FACE_AMBIGUOUS_MARGIN
        return {'code': self.code_version, 'face': model_identity(self.organizer.system),
                'shot': model_identity(self.organizer.shot_classifier),
                'embedding': model_identity(self.organizer.embedder),
                'role': model_identity(self.organizer.role_classifier) if self.params['classify_role'] else None,
                'face_settings': [self.organizer.face_threshold, FACE_IMAGE_MAX_SIDE, self.organizer.face_margin],
                'identity_registry': self.organizer.identity_registry_hash}

    def _verify(self, image):
        from core_modules.organize.identity_registry import registry_digest
        path = self.params.get('identity_registry_file')
        if path and registry_digest(path) != self.params['identity_registry_sha256']:
            raise ValueError('identity_registry_content_changed')
        if path and self.organizer.identity_registry_hash != self.params['identity_registry_sha256']:
            raise ValueError('identity_registry_execution_mismatch')
        self.organizer.verify_identity_registry()
        if fingerprint(image) != self.params['input_hashes'][image]:
            raise ValueError('source_content_changed')

    def process(self, item):
        image = item['image']
        self._verify(image)
        if item['stage'] == 'quality':
            img = cv2.imread(image)
            if img is None:
                raise ValueError('image_read_failed')
            dedup = BurstDeduplicator(self.organizer.shot_classifier)
            row = {'image': image, 'stage': 'quality', 'quality': image_quality(img),
                   'shot_type': dedup._shot_type(img, os.path.basename(image))}
        else:
            if gallery_hashes(self.params['db_root']) != self.params['gallery_hashes']:
                raise ValueError('face_gallery_content_changed')
            row = self.organizer.describe_image(image, classify_role=self.params['classify_role'], strict=True)
            if row.get('error'):
                raise ValueError(row['error'])
            row.update(image=image, stage='recognition')
            if self.organizer.embedder is not None:
                from core_modules.organize.feature_cache import FeatureService
                features = FeatureService(self.organizer.embedder, cache_dir=self.organizer.feature_cache_dir)
                vector = features.images([image]).get(image)
                # Preserve the pipeline's feature fallback, with a visible cause.
                row['embedding'] = vector.tolist() if vector is not None else None
                if vector is None:
                    row['feature_error'] = features.image_errors.get(image, 'unavailable')
        self._verify(image)
        return row

    def advance(self):
        if self.context.checkpoint('selection') is not None:
            return False
        self.context.check()
        quality = {row['image']: row for row in self.context.results()}
        groups = BurstDeduplicator().group_bursts(list(self.params['input_hashes']))
        selected, reports = [], []
        for index, (key, members) in enumerate(groups):
            buckets = {}
            for image in members:
                row = quality[image]
                if not row.get('error'):
                    buckets.setdefault(row['shot_type'], []).append(row)
            keep = set()
            for bucket in buckets.values():
                bucket.sort(key=lambda row: -row['quality']['score'])
                keep.update(row['image'] for row in bucket[:self.params['keep_per_bucket']])
            selected.extend(image for image in members if image in keep)
            reports.append({'group': index + 1, 'key': key, 'size': len(members), 'members': [
                {'name': os.path.basename(image), 'quality': quality[image].get('quality'),
                 'shot_type': quality[image].get('shot_type'), 'kept': image in keep,
                 **({'error': quality[image]['error']} if quality[image].get('error') else {})}
                for image in members]})
        self.context.advance('selection', {'selected': selected, 'groups': reports},
                             [{'image': image, 'stage': 'recognition'} for image in selected])
        return True

    def finish(self, rows):
        self.organizer.verify_identity_registry()
        if self.params['input_hashes']:
            self._verify(next(iter(self.params['input_hashes'])))
        if gallery_hashes(self.params['db_root']) != self.params['gallery_hashes']:
            raise ValueError('face_gallery_content_changed')
        plan = self.context.checkpoint('selection')
        selected = plan['selected']
        records = {row['image']: row for row in rows if row.get('stage') == 'recognition'}
        for row in rows:
            self.context.check()
            if not row.get('error'):
                self._verify(row['image'])
        embeddings = {image: records[image]['embedding'] for image in selected
                      if records[image].get('embedding') is not None}
        scenes = split_scenes(selected, gap_seconds=self.params['gap_seconds'],
                              embeddings=embeddings if self.organizer.embedder is not None else None,
                              strategy=self.params.get('scene_strategy', 'legacy'),
                              change_config=self.params.get('scene_change_config'),
                              order_trusted=self.params.get('scene_order_trusted', False))
        if self.params.get('scene_labels'):
            labels = self.params['scene_labels']
            scenes = {image: labels.get(scene, scene) for image, scene in scenes.items()}
        evidence_stats = None
        if self.params.get('scene_evidence'):
            scenes, evidence_stats = apply_scene_evidence(scenes, self.params['scene_evidence'])
        output = Path(self.params['output_dir'])
        successful, copied, reused = [], 0, 0
        failures = [row for row in rows if row.get('error')]
        for seq, image in enumerate(selected, 1):
            self.context.check()
            record = dict(records[image])
            record.pop('embedding', None)
            if record.get('error'):
                continue
            name = self.organizer.organized_name(image, scenes.get(image, 'scene-00'), record, seq)
            for directory, filename in (('best', os.path.basename(image)), ('organized', name)):
                self.context.check()
                new = copy_verified(image, output / directory / filename, self.params['input_hashes'][image])
                copied += int(new)
                reused += int(not new)
            record['organized_name'] = name
            successful.append(record)
        total = len(self.params['input_hashes'])
        quality_failed = sum(row.get('stage') == 'quality' and bool(row.get('error')) for row in rows)
        discarded = total - len(selected) - quality_failed
        dedup_report = {'input_dir': self.params['input_dir'], 'total': total, 'kept': len(selected),
                        'discarded': discarded, 'failed': quality_failed,
                        'n_groups': len(plan['groups']), 'groups': plan['groups']}
        report = {'input_dir': self.params['input_dir'], 'output_dir': str(output),
                  'db_root': self.params['db_root'], 'total_input': total,
                  'kept': len(successful), 'selected': len(selected), 'discarded': discarded,
                  'failed': len(failures), 'skipped': discarded, 'preserved': total,
                  'copied': len(successful), 'files_copied_this_attempt': copied,
                  'files_reused_this_attempt': reused, 'n_scenes': len(set(scenes.values())),
                  'scene_evidence': evidence_stats, 'records': successful, 'errors': failures,
                  'scene_strategy': self.params.get('scene_strategy', 'legacy'),
                  'scene_change_config': self.params.get('scene_change_config'),
                  'scene_order_trusted': self.params.get('scene_order_trusted', False),
                  'execution_contract': self.context.checkpoint('execution_contract'),
                  'burst': {'n_groups': len(plan['groups']), 'kept': len(selected), 'discarded': discarded}}
        self.context.check()
        atomic_json(output / 'dedup_report.json', dedup_report)
        self.context.check()
        atomic_json(output / 'organize_report.json', report)
        return {key: report[key] for key in ('total_input', 'kept', 'discarded', 'failed', 'copied', 'n_scenes', 'db_root')} | {
            'output_dir': str(output), 'report_path': str(output / 'organize_report.json')}
