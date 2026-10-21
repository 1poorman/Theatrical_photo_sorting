"""Checkpointed annotated-photo anchors and repeatable, verified publication."""
import base64
import hashlib
from pathlib import Path
import tempfile

import cv2
import numpy as np

from core_modules.organize import face_db_builder as rules
from core_modules.organize.face_db_builder import FaceDBBuilder
from core_modules.organize.filename_parser import parse_filename
from core_modules.organize.task_artifacts import atomic_json, copy_verified, fingerprint, face_database_images
from core_modules.organize.task_queue import TaskVersionConflict
from core_modules.organize.task_versions import execution_code, model_identity


class FaceDatabaseTask:
    def __init__(self, params, system):
        self.params = params
        self.builder = FaceDBBuilder(system, params['db_root'], params['max_anchors'])
        self.context = None
        self.code = execution_code([
            'app/organize_tasks.py', 'core_modules/organize/face_database_task.py',
            'core_modules/organize/face_db_builder.py', 'core_modules/organize/filename_parser.py',
            'core_modules/tools/image_io.py', 'core_modules/organize/task_artifacts.py',
            'core_modules/face_recognition/arcface/arcface_onnx.py',
            'core_modules/face_recognition/scrfd/scrfd_det.py'])

    def model_contract(self):
        return {'code': self.code, 'face': model_identity(self.builder.system)}

    def verify(self, image):
        self.context.check()
        if fingerprint(image, self.context.check) != self.params['input_hashes'][image]:
            raise ValueError('source_content_changed')

    def feature(self, img, face=None):
        self.context.check()
        feature, aligned = self.builder._face_feature(img, face)
        if feature is None or aligned is None or not np.isfinite(feature).all() or np.linalg.norm(feature) <= 0:
            raise ValueError('invalid_face_feature')
        return np.asarray(feature, dtype=np.float32), aligned

    def process(self, item):
        path = item['image']
        self.verify(path)
        if item['stage'] == 'gallery_snapshot':
            row = {'image': path, 'stage': 'gallery_snapshot'}
        elif item['stage'] == 'gallery':
            img = cv2.imread(path)
            if img is None:
                raise ValueError('gallery_image_read_failed')
            feature, _ = self.feature(img)
            row = {'image': path, 'stage': 'gallery', 'person': item['person'], 'feature': feature.tolist()}
        else:
            info = parse_filename(Path(path).name, done_style=item['done_style'])
            people = info['people']
            row = {'image': path, 'stage': 'anchors', 'people': people, 'faces': [], 'reason': None}
            if not 1 <= len(people) <= rules.MAX_PEOPLE_IN_GROUP:
                row['reason'] = 'unlabelled'
            elif any(word in Path(path).name for word in rules.SKIP_KEYWORDS):
                row['reason'] = 'group'
            else:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError('image_read_failed')
                if max(img.shape[:2]) > 1920:
                    from core_modules.tools.image_io import resize_max_side
                    img = resize_max_side(img, 1920)
                faces = self.builder._detect_faces(img)
                if not faces:
                    row['reason'] = 'no_face'
                elif len(people) == 1 and len(faces) > 1 and faces[1].width * faces[1].height >= \
                        rules.AMBIGUOUS_AREA_RATIO * faces[0].width * faces[0].height:
                    row['reason'] = 'ambiguous'
                elif len(people) > 1 and len(faces) != len(people):
                    row['reason'] = 'face_count_mismatch'
                else:
                    for face in faces[:1] if len(people) == 1 else faces:
                        raw, aligned = self.feature(img, face)
                        ok, encoded = cv2.imencode('.jpg', aligned)
                        if not ok:
                            raise ValueError('anchor_encoding_failed')
                        # Next propagation round reads published JPEGs. Cache that
                        # exact feature too, so planning never re-runs a model.
                        stored, _ = self.feature(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
                        data = encoded.tobytes()
                        row['faces'].append({'raw': raw.tolist(), 'stored': stored.tolist(),
                                             'jpeg': base64.b64encode(data).decode('ascii'),
                                             'sha256': hashlib.sha256(data).hexdigest()})
        self.verify(path)
        return row

    def initial_gallery(self):
        root = Path(self.params['db_root'])
        return {str(Path(path).relative_to(root)): self.params['input_hashes'][path]
                for path in self.params['gallery_paths']}

    def verify_gallery(self, anchors=()):
        before = self.initial_gallery()
        root = Path(self.params['db_root'])
        current = {str(path.relative_to(root)): fingerprint(path, self.context.check)
                   for path in face_database_images(root)}
        allowed = {anchor['relative']: anchor['sha256'] for anchor in anchors}
        if any(current.get(path) != digest for path, digest in before.items()) or any(
                path not in before and allowed.get(path) != digest for path, digest in current.items()):
            raise TaskVersionConflict('face_gallery_content_changed; 请创建新任务')

    def advance(self):
        if self.context.checkpoint('anchor_plan') is not None:
            return False
        self.context.check()
        rows = self.context.results()
        if any(row.get('error') and row['image'] in self.params['gallery_paths'] for row in rows):
            raise ValueError('initial_gallery_feature_failed')
        self.verify_gallery()
        library = {}
        for row in rows:
            if row.get('stage') == 'gallery':
                library.setdefault(row['person'], []).append((np.asarray(row['feature'], dtype=np.float32), row['image']))
        existing_library = {person: list(features) for person, features in library.items()}
        counts = dict(self.params['gallery_counts'])
        by_image = {row['image']: row for row in rows}
        photos = [by_image[path] for path in self.params['photo_paths']]
        anchors = {}
        stats = {'scanned': len(photos), 'candidates': 0, 'added': 0, 'reused': 0,
                 'skipped_multi': 0, 'skipped_group': 0, 'skipped_ambiguous': 0,
                 'skipped_no_face': 0, 'skipped_max': 0, 'skipped_inconsistent': 0,
                 'added_persons': [], 'failed': sum(bool(row.get('error')) for row in photos),
                 'dry_run': self.params['dry_run']}
        reused = set()
        def add(row, face, person, check_library=None):
            if not person or person in ('.', '..') or any(char in person for char in ('/', '\\', '\x00')):
                raise ValueError('invalid_anchor_person_directory')
            relative = str(Path(person) / (Path(row['image']).stem + '_face.jpg'))
            if relative in anchors:
                if anchors[relative]['sha256'] != face['sha256']:
                    raise ValueError('anchor_output_conflict: ' + relative)
                return False
            original = self.initial_gallery().get(relative)
            if original is not None:
                if original != face['sha256']:
                    raise ValueError('anchor_output_conflict: ' + relative)
                reused.add(relative)
                return False
            raw = np.asarray(face['raw'], dtype=np.float32)
            if check_library is not None and not self.builder.anchor_consistent(raw, person, check_library):
                stats['skipped_inconsistent'] += 1
                return False
            anchors[relative] = dict(face, image=row['image'], person=person, relative=relative)
            counts[person] = counts.get(person, 0) + 1
            if check_library is not None:
                check_library.setdefault(person, []).append((raw, relative))
            return True
        for row in photos:
            if row.get('error'):
                continue
            people = row['people']
            if len(people) != 1:
                stats['skipped_multi'] += 1
                continue
            if row['reason'] == 'group':
                stats['skipped_group'] += 1
                continue
            person = self.builder._person_dir(*people[0])
            if counts.get(person, 0) >= self.params['max_anchors']:
                stats['skipped_max'] += 1
                continue
            stats['candidates'] += 1
            if row['reason']:
                stats['skipped_ambiguous' if row['reason'] == 'ambiguous' else 'skipped_no_face'] += 1
                continue
            add(row, row['faces'][0], person)
        for _ in range(rules.PROPAGATE_ROUNDS):
            library = {person: list(features) for person, features in existing_library.items()}
            for anchor in anchors.values():
                library.setdefault(anchor['person'], []).append((np.asarray(anchor['stored'], dtype=np.float32), anchor['relative']))
            if not library:
                break
            propagated = 0
            for row in photos:
                self.context.check()
                if row.get('error') or row['reason'] or not 2 <= len(row['people']) <= rules.MAX_PEOPLE_IN_GROUP:
                    continue
                features = [np.asarray(face['raw'], dtype=np.float32) for face in row['faces']]
                choice = self.builder.propagation_choice(row['people'], features, counts, library)
                if choice is not None:
                    index, person = choice
                    propagated += int(add(row, row['faces'][index], person, library))
            if not propagated:
                break
        stats['added'] = len(anchors)
        stats['reused'] = len(reused)
        stats['added_persons'] = sorted({anchor['person'] for anchor in anchors.values()})
        self.context.advance('anchor_plan', {'anchors': list(anchors.values()), 'stats': stats}, [])
        return True

    def finish(self, rows):
        plan = self.context.checkpoint('anchor_plan')
        anchors = plan['anchors']
        for row in rows:
            if not row.get('error') and row['image'] not in self.params['gallery_paths']:
                self.verify(row['image'])
        self.verify_gallery(anchors if not self.params['dry_run'] else ())
        copied, reused = 0, 0
        if not self.params['dry_run']:
            for anchor in anchors:
                self.context.check()
                self.verify(anchor['image'])
                self.verify_gallery(anchors)
                data = base64.b64decode(anchor['jpeg'], validate=True)
                with tempfile.NamedTemporaryFile() as temporary:
                    temporary.write(data)
                    temporary.flush()
                    new = copy_verified(temporary.name, Path(self.params['db_root']) / anchor['relative'], anchor['sha256'])
                copied += int(new)
                reused += int(not new)
        self.verify_gallery(anchors if not self.params['dry_run'] else ())
        result = dict(plan['stats'], db_root=self.params['db_root'], files_copied_this_attempt=copied,
                      files_reused_this_attempt=reused, errors=[row for row in rows if row.get('error')],
                      execution_contract=self.context.checkpoint('execution_contract'))
        if self.params.get('output_dir'):
            self.context.check()
            path = Path(self.params['output_dir']) / 'face_database_report.json'
            atomic_json(path, result)
            result['report_path'] = str(path)
        return result
