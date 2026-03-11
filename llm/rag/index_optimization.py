from pymilvus import \
    MilvusClient, AnnSearchRequest, Function, FunctionType
from typing import Tuple, Iterator
from embedding_model import get_bge_embeddings_single, BGE_EMBEDDING_DIM
from rag_config import *

def initialize_db():
    client = MilvusClient(
        url=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    client.use_database(DB_NAME)
    return client

class FullTextRetrieval:
    def __init__(self):
        self.client = initialize_db() 
    
    def query(self, queries) -> list:
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=queries,
            anns_field='sparse_vector',
            limit=5,
            output_fields=['content', 'metadata']
        )
        sparse_results = results[0]
        # for result in sparse_results:
            # yield result['distance'], result['entity']['content'] 
        for i, result in enumerate(sparse_results):
            print(f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}")
        return list(map(lambda result: result['entity']['content'], sparse_results))  

class HybridTextRetrieval:
    def __init__(self):
        self.client = initialize_db()
        self.ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                'reranker': 'rrf',
                "k": 100
            }
        )         

    def query(self, query_text) -> list:
        query_dense_vector = get_bge_embeddings_single(query_text)
        search_param_1 = {
            'data': [query_dense_vector],
            'anns_field': "dense_vector",
            'param': {'nprobe': 10},
            'limit': 2
        }
        request1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            'data': [query_text],
            'anns_field': "sparse_vector",
            'param': {'nprobe': 10},
            'limit': 2
        }
        request2 = AnnSearchRequest(**search_param_2)

        reqs = [request1, request2]

        res = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=reqs,
            ranker=self.ranker,
            limit=2,
            output_fields=['content', 'metadata']
        )

        for hits in res:
            print('TopK results:')
            for hit in hits:
                print(hit)

if __name__ == '__main__':
    # ret = FullTextRetrieval()
    ret = HybridTextRetrieval() 
    ret.query(['graph neural-network'])
    # ret.query('Graph')
