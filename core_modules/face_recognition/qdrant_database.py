"""Qdrant face anchors, preserving legacy exact cosine and deduplication semantics."""
from datetime import datetime, timezone
import hashlib

from config.base import QDRANT_PREFIX
from core_modules.vector_store import QdrantStore, normalized, point_id, match_filter
from core_modules.face_recognition.database import FaceDatabase


def collection_name(model_signature):
    if not model_signature:
        raise ValueError('face_model_signature_required')
    return QDRANT_PREFIX + '_faces_' + hashlib.sha256(model_signature.encode()).hexdigest()[:20]


class QdrantFaceDatabase:
    _dedupe_hits = FaceDatabase._dedupe_hits

    def __init__(self, model_signature, store=None):
        self.store = store or QdrantStore()
        self.index_name = collection_name(model_signature)
        self.model_signature = model_signature
        self.store.ensure(self.index_name, 512, ('person_id',))

    def _point(self, person_id, person_name, embedding, image_path, metadata=None):
        payload = {'person_id': person_id, 'person_name': person_name, 'image_path': image_path,
                   'model_signature': self.model_signature, 'metadata': metadata or {},
                   'timestamp': datetime.now(timezone.utc).isoformat()}
        return {'id': point_id(str(person_id) + '\n' + str(image_path)),
                'vector': normalized(embedding, 512), 'payload': payload}

    def add_face(self, person_id, person_name, embedding, image_path, metadata=None):
        point = self._point(person_id, person_name, embedding, image_path, metadata)
        self.store.upsert(self.index_name, [point])
        return dict(point['payload'], face_embedding=point['vector'])

    def batch_add_faces(self, faces_data):
        count, batch = 0, []
        for face in faces_data:
            batch.append(self._point(face['person_id'], face['person_name'], face['embedding'],
                                     face['image_path'], face.get('metadata')))
            if len(batch) == 128:
                self.store.upsert(self.index_name, batch)
                count += len(batch)
                batch = []
        self.store.upsert(self.index_name, batch)
        return count + len(batch)

    def _matches(self, hits, top_k, threshold):
        response = {'hits': {'hits': [{'_source': hit['payload'], '_score': max(0.0, float(hit['score']))}
                                      for hit in hits]}}
        return self._dedupe_hits(response, top_k, threshold)

    def search_face(self, query_embedding, top_k=5, threshold=.7):
        hits = self.store.query(self.index_name, normalized(query_embedding, 512), limit=top_k, exact=True)
        return self._matches(hits, top_k, threshold)

    def search_faces_batch(self, query_embeddings, top_k=5, threshold=.7):
        vectors = [normalized(vector, 512) for vector in query_embeddings]
        hits = self.store.query_batch(self.index_name, vectors, limit=top_k, exact=True)
        return [self._matches(row, top_k, threshold) for row in hits]

    def delete_by_person_id(self, person_id):
        self.store.delete_points(self.index_name, filters=match_filter({'person_id': person_id}))

    def check_archive_exists(self, guid):
        exists = self.store.count(self.index_name, match_filter({'person_id': guid})) > 0
        if exists:
            self.delete_by_person_id(guid)
        return exists

    def knn_search(self, query_vector, k=10, num_candidates=100):
        hits = self.store.query(self.index_name, normalized(query_vector, 512), limit=k,
                                hnsw_ef=max(k, num_candidates))
        return [dict(hit['payload'], score=float(hit['score'])) for hit in hits]
