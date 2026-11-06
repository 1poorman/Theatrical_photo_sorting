"""Small, versioned Qdrant REST adapter; vectors never accumulate in the client.

Targets Qdrant 1.20. Uses acknowledged writes, explicit collections and raw cosine
scores. No silent fallback to a stale local index when the service is unavailable.
"""
import json
import math
import re
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config.base import QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, QDRANT_HNSW_EF


class VectorStoreError(RuntimeError):
    pass


def normalized(vector, dimension=None):
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or (dimension is not None and len(array) != dimension):
        raise ValueError('vector_dimension_mismatch')
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or not math.isfinite(norm) or norm <= 0:
        raise ValueError('vector_must_be_finite_and_nonzero')
    return (array / norm).tolist()


def point_id(value):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def match_filter(values):
    if not values:
        return None
    if not isinstance(values, dict):
        raise ValueError('filter_must_be_object')
    return {'must': [{'key': key, 'match': {'value': value}} for key, value in values.items()]}


class QdrantStore:
    def __init__(self, url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method, path, body=None, allow_missing=False):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['api-key'] = self.api_key
        data = json.dumps(body, allow_nan=False, separators=(',', ':')).encode() if body is not None else None
        try:
            with urlopen(Request(self.url + path, data=data, headers=headers, method=method), timeout=self.timeout) as response:
                value = json.load(response)
        except HTTPError as exc:
            if allow_missing and exc.code == 404:
                return None
            # Never include endpoint credentials or arbitrary server response bodies.
            raise VectorStoreError(f'qdrant_http_{exc.code}: {method} {path}') from None
        except (URLError, TimeoutError, OSError) as exc:
            raise VectorStoreError('qdrant_unavailable: ' + type(exc).__name__) from None
        if value.get('status') != 'ok':
            raise VectorStoreError('qdrant_request_failed')
        return value.get('result')

    @staticmethod
    def path(collection):
        if not isinstance(collection, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', collection):
            raise ValueError('invalid_collection_name')
        return '/collections/' + collection

    def info(self, collection, allow_missing=False):
        return self.request('GET', self.path(collection), allow_missing=allow_missing)

    def create(self, collection, dimension, indexed_fields=(), indexing_threshold=1000, metadata=None):
        if dimension <= 0:
            raise ValueError('dimension_must_be_positive')
        self.request('PUT', self.path(collection), {
            'vectors': {'size': dimension, 'distance': 'Cosine', 'memory': 'cold'},
            'hnsw_config': {'m': 16, 'ef_construct': 100, 'memory': 'cold'},
            'optimizers_config': {'indexing_threshold': indexing_threshold, 'max_optimization_threads': 2},
            'wal_config': {'wal_capacity_mb': 4}, 'on_disk_payload': True,
            'metadata': metadata or {},
        })
        for field in indexed_fields:
            self.request('PUT', self.path(collection) + '/index?wait=true',
                         {'field_name': field, 'field_schema': 'keyword'})

    def ensure(self, collection, dimension, indexed_fields=()):
        info = self.info(collection, allow_missing=True)
        if info is None:
            try:
                self.create(collection, dimension, indexed_fields)
            except VectorStoreError:
                # Another initializer may have created the same collection.
                if self.info(collection, allow_missing=True) is None:
                    raise
            info = self.info(collection)
        config = info['config']['params']['vectors']
        if config.get('size') != dimension or config.get('distance') != 'Cosine':
            raise ValueError('collection_vector_space_mismatch')
        return info

    def upsert(self, collection, points):
        if not points:
            return
        self.request('PUT', self.path(collection) + '/points?wait=true', {'points': points})

    def query(self, collection, vector, limit=5, filters=None, exact=False, with_vectors=False, hnsw_ef=QDRANT_HNSW_EF):
        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError('top_k_must_be_between_1_and_10000')
        body = {'query': normalized(vector), 'limit': limit,
                'with_payload': True, 'with_vector': with_vectors,
                'params': {'exact': bool(exact), 'hnsw_ef': hnsw_ef}}
        if filters:
            body['filter'] = filters
        return self.request('POST', self.path(collection) + '/points/query', body)['points']

    def query_batch(self, collection, vectors, limit=5, exact=True):
        requests = [{'query': normalized(vector), 'limit': limit,
                     'with_payload': True, 'with_vector': False, 'params': {'exact': exact}}
                    for vector in vectors]
        if not requests:
            return []
        result = self.request('POST', self.path(collection) + '/points/query/batch', {'searches': requests})
        return [item['points'] for item in result]

    def count(self, collection, filters=None):
        body = {'exact': True}
        if filters:
            body['filter'] = filters
        return self.request('POST', self.path(collection) + '/points/count', body)['count']

    def retrieve(self, collection, ids, with_vectors=False):
        if not ids:
            return []
        return self.request('POST', self.path(collection) + '/points',
                            {'ids': ids, 'with_payload': True, 'with_vector': with_vectors})

    def delete_points(self, collection, ids=None, filters=None):
        if ids is None and not filters:
            raise ValueError('explicit_delete_selector_required')
        if ids == []:
            return
        self.request('POST', self.path(collection) + '/points/delete?wait=true',
                     {'points': ids} if ids is not None else {'filter': filters})

    def scroll(self, collection, batch_size=128, with_vectors=False, filters=None):
        offset = None
        while True:
            body = {'limit': batch_size, 'with_payload': True, 'with_vector': with_vectors}
            if offset is not None:
                body['offset'] = offset
            if filters:
                body['filter'] = filters
            page = self.request('POST', self.path(collection) + '/points/scroll', body)
            yield from page['points']
            offset = page.get('next_page_offset')
            if offset is None:
                return

    def delete_collection(self, collection):
        self.request('DELETE', self.path(collection))
