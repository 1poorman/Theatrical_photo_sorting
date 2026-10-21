"""Durable image features, grouping and preview publication for face scanning."""
import base64
import hashlib
from pathlib import Path
import tempfile

import cv2
import numpy as np

from core_modules.organize.face_cluster import FaceClusterBuilder, CLUSTER_COLLAGE_MAX, STATE_FILE
from core_modules.organize.feature_cache import signature
from core_modules.organize.task_artifacts import atomic_json, copy_verified, fingerprint
from core_modules.organize.task_versions import execution_code, model_identity


class FaceClusterTask:
    def __init__(self, params, system):
        self.params = params
        self.builder = FaceClusterBuilder(system)
        self.context = None
        self.code = execution_code([
            'app/organize_tasks.py', 'core_modules/organize/face_cluster_task.py',
            'core_modules/organize/face_cluster.py', 'core_modules/tools/image_io.py',
            'core_modules/face_recognition/arcface/arcface_onnx.py',
            'core_modules/face_recognition/scrfd/scrfd_det.py'])

    def model_contract(self):
        from importlib.metadata import version
        return {'code': self.code, 'face': model_identity(self.builder.system),
                'sklearn': version('scikit-learn')}

    def verify(self, image):
        self.context.check()
        if fingerprint(image, self.context.check) != self.params['input_hashes'][image]:
            raise ValueError('source_content_changed')

    def process(self, item):
        if item['stage'] == 'faces':
            self.verify(item['image'])
            members = self.builder.scan_image(item['image'], self.params['min_face_area'], self.context.check)
            self.verify(item['image'])
            return {'image': item['image'], 'stage': 'faces', 'members': members}
        cluster = next(cluster for cluster in self.context.checkpoint('clusters') if cluster['id'] == item['cluster_id'])
        selected = cluster['members'][:CLUSTER_COLLAGE_MAX]
        cell, cols = 160, 6
        canvas = np.full((((len(selected) + cols - 1) // cols) * cell, cols * cell, 3), 245, dtype=np.uint8)
        for index, member in enumerate(selected):
            self.verify(member['img_path'])
            face = self.builder.aligned_member(member)
            self.verify(member['img_path'])
            row, column = divmod(index, cols)
            canvas[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = face
        ok, jpeg = cv2.imencode('.jpg', canvas)
        if not ok:
            raise ValueError('preview_encoding_failed')
        return {'image': item['image'], 'stage': 'preview', 'cluster_id': cluster['id'],
                'jpeg': base64.b64encode(jpeg).decode('ascii')}

    def advance(self):
        if self.context.checkpoint('clusters') is not None:
            return False
        self.context.check()
        members = [member for row in self.context.results() if not row.get('error') for member in row['members']]
        clusters = self.builder.cluster_members(members, self.params['dist_threshold'])
        self.context.check()
        self.context.advance('clusters', clusters, [{'image': cluster['members'][0]['img_path'],
                             'stage': 'preview', 'cluster_id': cluster['id']} for cluster in clusters])
        return True

    def finish(self, rows):
        plan = self.context.checkpoint('clusters')
        failures = [row for row in rows if row.get('error')]
        if any(row.get('stage') == 'preview' for row in failures):
            raise ValueError('cluster_preview_failed; 请核对源图后创建新任务')
        for row in rows:
            if row.get('stage') == 'faces' and not row.get('error'):
                self.verify(row['image'])
        output = Path(self.params['output_dir'])
        scan_id = signature({'hashes': self.params['input_hashes'],
                             'threshold': self.params['dist_threshold'], 'area': self.params['min_face_area'],
                             'version': self.context.checkpoint('execution_contract')})
        path = output / STATE_FILE
        previews = {row['cluster_id']: row for row in rows if row.get('stage') == 'preview'}
        if path.exists():
            state = self.builder.load_state(path)
            if state.get('scan_id') != scan_id:
                raise ValueError('cluster_output_conflict; 请选择新的输出目录')
            for cluster in state['clusters']:
                expected = base64.b64decode(previews[cluster['id']]['jpeg'], validate=True)
                preview = Path(cluster['preview'])
                if not preview.is_file() or fingerprint(preview) != hashlib.sha256(expected).hexdigest():
                    raise ValueError('cluster_preview_content_conflict')
            return self.summary(state, path)
        for cluster in plan:
            self.context.check()
            preview = output / 'previews' / (cluster['id'] + '.jpg')
            data = base64.b64decode(previews[cluster['id']]['jpeg'], validate=True)
            with tempfile.NamedTemporaryFile() as temporary:
                temporary.write(data)
                temporary.flush()
                copy_verified(temporary.name, preview, hashlib.sha256(data).hexdigest())
            cluster['preview'] = str(preview)
        state = {'input_dir': self.params['input_dir'], 'output_dir': str(output), 'scan_id': scan_id,
                 'dist_threshold': self.params['dist_threshold'], 'min_face_area': self.params['min_face_area'],
                 'n_images': len(self.params['input_hashes']), 'n_faces': sum(cluster['size'] for cluster in plan),
                 'clusters': plan, 'failed': len(failures), 'errors': failures,
                 'input_hashes': self.params['input_hashes'],
                 'execution_contract': self.context.checkpoint('execution_contract')}
        # A single valid face retains the legacy empty-cluster behavior.
        if not plan:
            state['n_faces'] = sum(len(row.get('members', [])) for row in rows)
        self.context.check()
        atomic_json(path, state)
        return self.summary(state, path)

    @staticmethod
    def summary(state, path):
        return {'state_path': str(path), 'n_images': state['n_images'], 'n_faces': state['n_faces'],
                'n_clusters': len(state['clusters']), 'failed': state.get('failed', 0),
                'n_suggest_manual': sum(cluster['suggest_manual'] for cluster in state['clusters'])}
