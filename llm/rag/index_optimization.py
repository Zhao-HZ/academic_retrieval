from pymilvus import MilvusClient
from typing import Tuple, Iterator
from rag_config import *

class Retrieval:
    def __init__(self):
        self.client = MilvusClient(
           ur=MILVUS_URI,
           token=MILVUS_TOKEN
        )  
        self.client.use_database(DB_NAME)
    
    def fullTextSearch(self, queries) -> list:
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
  
    def denseSearch(self, )
   
    def hybridSearch(self):
        ...
    
if __name__ == '__main__':
    ret = Retrieval()
    ret.fullTextSearch(['graph neural-network'])