# Added imports for required modules
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
from datetime import datetime
import json
from config.base import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASSWORD 

class FaceDatabase:
    def __init__(self, host=ELASTIC_URL,
     user = ELASTIC_USER, password = ELASTIC_PASSWORD, index_name='face_database_512'):
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
    
    # def search_face(self, query_embedding, top_k=5, threshold=0.7):
    #     """
    #     搜索最相似的人脸
    #     """
    #     script_query = {
    #         "script_score": {
    #             "query": {"match_all": {}},
    #             "script": {
    #                 "source": """
    #                 double cosine = cosineSimilarity(params.query_vector, 'face_embedding');
    #                 // Apply transformation to increase discrimination
    #                 // Using power function to widen the gap between high and low similarities
    #                 return Math.pow((cosine + 1.0) / 2.0, 2);  // Square the normalized cosine similarity
    #                 """,
    #                 "params": {"query_vector": query_embedding.tolist()}
    #             }
    #         }
    #     }
        
    #     response = self.es.search(
    #         index=self.index_name,
    #         body={
    #             "size": top_k,
    #             "query": script_query,
    #             "_source": ["person_id", "person_name", "image_path"]
    #         }
    #     )
        
    #     results = []
    #     for hit in response['hits']['hits']:
    #         # Elasticsearch cosineSimilarity返回值范围是[-1, 1]，经过+1后是[0, 2]
    #         # 所以需要除以2转换到[0, 1]范围
    #         score = hit['_score'] 
            
    #         # 调试输出
    #         print(f"DEBUG: Face match score: {score}, threshold: {threshold}")
            
    #         if score >= threshold:
    #             result = {
    #                 "person_id": hit['_source']['person_id'],
    #                 "person_name": hit['_source']['person_name'],
    #                 "score": score,
    #                 "image_path": hit['_source'].get('image_path')
    #             }
    #             results.append(result)
        
    #     # 按分数降序排列
    #     results.sort(key=lambda x: x['score'], reverse=True)
    #     return results
    
    # In database.py, replace the existing search_face method with this enhanced version:

    def search_face(self, query_embedding, top_k=5, threshold=0.7):
        """
        搜索最相似的人脸，并自动去重，为每个人只返回最佳匹配
        """
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                    double cosine = cosineSimilarity(params.query_vector, 'face_embedding');
                    // Apply transformation to increase discrimination
                    // Using power function to widen the gap between high and low similarities
                    return Math.pow((cosine + 1.0) / 2.0, 2);  // Square the normalized cosine similarity
                    """,
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
        
        # Group results by person name and keep only the best score for each person
        person_matches = {}
        
        for hit in response['hits']['hits']:
            # Elasticsearch cosineSimilarity返回值范围是[-1, 1]，经过处理后是[0, 1]
            score = hit['_score'] 
            
            # Debug output
            print(f"DEBUG: Face match score: {score}, threshold: {threshold}")
            
            if score >= threshold:
                person_name = hit['_source']['person_name']
                
                # Create result object
                result = {
                    "person_id": hit['_source']['person_id'],
                    "person_name": person_name,
                    "score": score,
                    "image_path": hit['_source'].get('image_path')
                }
                
                # Keep only the best match for each person
                if person_name not in person_matches or score > person_matches[person_name]['score']:
                    person_matches[person_name] = result
        
        # Convert to list and sort by score
        results = sorted(person_matches.values(), key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k unique persons
        results = results[:top_k]
        return results
        
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
    