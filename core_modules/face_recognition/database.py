# Added imports for required modules
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
from datetime import datetime
import json
from config.base import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASSWORD, ELASTIC_FACE_INDEX
from core_modules.tools.logger import get_app_logger

logger = get_app_logger()

class FaceDatabase:
    def __init__(self, host=ELASTIC_URL,
     user = ELASTIC_USER, password = ELASTIC_PASSWORD, index_name=ELASTIC_FACE_INDEX):
        # Handle authentication if credentials are provided
        if user and password:
            self.es = Elasticsearch(hosts=host, basic_auth=(user, password))
        else:
            self.es = Elasticsearch(hosts=host)
        self.index_name = index_name
        
        # Ensure index exists
        self._ensure_index_exists()
                
    
    def _ensure_index_exists(self):
        """Ensure the face index exists, create if it doesn't"""
        if not self.es.indices.exists(index=self.index_name):
            self.create_index()
    
    # In database.py, update the create_index method:
    def create_index(self):
        """创建人脸索引"""
        # 清空现有索引
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        
        # Create the index with correct configuration
        mapping = {
            "mappings": {
                "properties": {
                    "person_id": {"type": "keyword"},
                    "person_name": {"type": "text"},
                    "face_embedding": {
                        "type": "dense_vector",
                        "dims": 512,  # Changed to match your actual model output
                        "index": True,
                        "similarity": "cosine"
                    },
                    "image_path": {"type": "keyword"},
                    "detection_info": {"type": "object"},
                    "timestamp": {"type": "date"},
                    "metadata": {"type": "object"}
                }
            },
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"Created new index {self.index_name} with 512 dimensions")
    
    def add_face(self, person_id, person_name, embedding, image_path, metadata=None):
        """Add a single face record"""
        doc = {
            "person_id": person_id,
            "person_name": person_name,
            "face_embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.es.index(index=self.index_name, body=doc)
        return doc
    
    def batch_add_faces(self, faces_data):
        """Batch add faces"""
        actions = []
        for face in faces_data:
            action = {
                "_index": self.index_name,
                "_source": {
                    "person_id": face["person_id"],
                    "person_name": face["person_name"],
                    "face_embedding": face["embedding"].tolist(),
                    "image_path": face["image_path"],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": face.get("metadata", {})
                }
            }
            actions.append(action)
        
        success, _ = bulk(self.es, actions)
        return success
    
    

    def search_face(self, query_embedding, top_k=5, threshold=0.7):
        """
        搜索最相似的人脸，并自动去重，为每个人只返回最佳匹配

        分数语义：直接返回余弦相似度。ES script_score 要求分数非负，
        故负余弦（不相似）截断为 0——正区间（人脸匹配的实际范围）保持原始余弦。
        
        """
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "return Math.max(0.0, cosineSimilarity(params.query_vector, 'face_embedding'));",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }

        response = self.es.search(
            index=self.index_name,
            body={
                "size": top_k,  # Request more results to allow for deduplication
                "query": script_query,
                "_source": ["person_id", "person_name", "image_path"]
            }
        )

        return self._dedupe_hits(response, top_k, threshold)

    def search_faces_batch(self, query_embeddings, top_k=5, threshold=0.7):
        """批量人脸检索：一次 msearch 完成多张脸查询（多脸图片省 N-1 次网络往返）。

        Args:
            query_embeddings: [N, 512] 已 L2 归一化的 embedding 列表
        Returns:
            list[list[dict]]: 与输入顺序对应的每张脸匹配结果
        """
        if not len(query_embeddings):
            return []

        def _q(vec):
            return {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "return Math.max(0.0, cosineSimilarity(params.query_vector, 'face_embedding'));",
                            "params": {"query_vector": np.asarray(vec).tolist()}
                        }
                    }
                },
                "_source": ["person_id", "person_name", "image_path"]
            }

        body = []
        for vec in query_embeddings:
            body.append({"index": self.index_name})
            body.append(_q(vec))

        try:
            responses = self.es.msearch(body=body, index=self.index_name)
        except Exception as e:
            logger.warning(f"msearch failed ({e}), falling back to per-face search")
            return [self.search_face(v, top_k=top_k, threshold=threshold)
                    for v in query_embeddings]

        out = []
        for resp in responses["responses"]:
            if "error" in resp:
                logger.warning(f"msearch item error: {resp['error'].get('type')}")
                out.append([])
                continue
            out.append(self._dedupe_hits(resp, top_k, threshold))
        return out

    def _dedupe_hits(self, response, top_k, threshold):
        """解析 ES 命中：按人去重保留最高分，过滤阈值以下，排序截断"""
        person_matches = {}

        for hit in response['hits']['hits']:
            # 分数即原始余弦相似度（[-1,1]），与 config 阈值语义一致
            score = hit['_score']

            logger.debug(f"Face match score: {score:.4f}, threshold: {threshold}")
            if score >= threshold:
                person_name = hit['_source']['person_name']

                result = {
                    "person_id": hit['_source']['person_id'],
                    "person_name": person_name,
                    "score": score,
                    "image_path": hit['_source'].get('image_path')
                }

                # 每人只保留最佳匹配
                if person_name not in person_matches or score > person_matches[person_name]['score']:
                    person_matches[person_name] = result

        results = sorted(person_matches.values(), key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def delete_by_person_id(self, person_id):
        """Delete all face records for a specific person"""
        query = {
            "query": {
                "term": {"person_id": person_id}
            }
        }
        self.es.delete_by_query(index=self.index_name, body=query)
    
    # New functions added based on elasticClient.py
    
    def check_archive_exists(self, guid):
        """
        Check if archive exists and delete if found
        """
        search_query = {
            "query": {"bool": {"must": [{"term": {"person_id": guid}}]}},
            "_source": ["person_id", "image_path"],
            "size": 10000
        }
        delete_query = {"query": {"bool": {"must": [{"term": {"person_id": guid}}]}}}
        
        res = self.es.search(index=self.index_name, body=search_query)
        if res["hits"]["total"]["value"] > 0:
            self.es.delete_by_query(index=self.index_name, body=delete_query)
            return True
        else:
            return False
    
    def knn_search(self, query_vector, k=10, num_candidates=100):
        """
        KNN vector search similar to elastic_ann_search
        """
        res = self.es.search(
            index=self.index_name,
            source=["person_id", "person_name", "image_path"],
            knn=[{
                "field": "face_embedding",
                "k": k,
                "num_candidates": num_candidates,
                "query_vector": query_vector
            }]
        )
        
        content = []
        for item in res["hits"]["hits"]:
            score = item["_score"]
            row = item["_source"]
            row["score"] = score
            content.append(row)
        return content
    