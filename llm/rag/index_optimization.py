from pymilvus import \
    MilvusClient, AnnSearchRequest, Function, FunctionType
from typing import Tuple, Iterator, List, Dict, Any, Optional
from llm.rag.embedding_model import get_openai_qwen_embeddings
from llm.llm_config import *

# Import BGE Reranker model
from sentence_transformers import CrossEncoder

# Singleton instance for reranker model
_reranker_model = None


def get_reranker_model(model_name: str = "BAAI/bge-reranker-large") -> CrossEncoder:
    """
    Get BGE Reranker model (singleton pattern).

    Args:
        model_name: Model name, default is bge-reranker-large

    Returns:
        CrossEncoder model instance
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading Reranker model: {model_name}")
        _reranker_model = CrossEncoder(model_name, max_length=512)
        print("Reranker model loaded successfully")
    return _reranker_model

def initialize_db():
    client = MilvusClient(
        url=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    client.use_database(DB_NAME)
    return client

def create_indexes(client):
    """Create indexes for the collection if they don't exist."""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25'
    )
    index_params.add_index(field_name='dense_vector', index_type='FLAT', metric_type='IP')
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

class FullTextRetrieval:
    def __init__(self):
        self.client = initialize_db()
        # Create indexes if not exist
        create_indexes(self.client)
        # Load collection into memory before searching
        self.client.load_collection(collection_name=COLLECTION_NAME)
    
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
        # Create indexes if not exist
        create_indexes(self.client)
        # Load collection into memory before searching
        self.client.load_collection(collection_name=COLLECTION_NAME)
        self.ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                'reranker': 'rrf',
                "k": 100
            }
        )         

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        use_bge_reranker: bool = False,
        rerank_top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the hybrid retrieval system.

        Args:
            query_text: The query string
            top_k: Number of results to retrieve
            use_bge_reranker: Whether to use BGE Reranker for re-ranking
            rerank_top_n: Number of results to return after re-ranking (default: top_k)

        Returns:
            List of result dictionaries with content, metadata, and score
        """
        # Get more candidates for reranking
        candidate_top_k = max(top_k * 3, 20) if use_bge_reranker else top_k
        if rerank_top_n is None:
            rerank_top_n = top_k

        # Generate dense vector for the query
        query_dense_vector = get_openai_qwen_embeddings([query_text])[0]

        # Dense vector search
        search_param_1 = {
            'data': [query_dense_vector],
            'anns_field': "dense_vector",
            'param': {'nprobe': 10},
            'limit': candidate_top_k
        }
        request1 = AnnSearchRequest(**search_param_1)

        # Sparse vector search (BM25)
        search_param_2 = {
            'data': [query_text],
            'anns_field': "sparse_vector",
            'param': {'nprobe': 10},
            'limit': candidate_top_k
        }
        request2 = AnnSearchRequest(**search_param_2)

        reqs = [request1, request2]

        res = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=reqs,
            ranker=self.ranker,
            limit=candidate_top_k,
            output_fields=['content', 'metadata']
        )

        # Extract candidates
        candidates = []
        for hits in res:
            for hit in hits:
                candidates.append({
                    'content': hit['entity']['content'],
                    'metadata': hit['entity'].get('metadata', {}),
                    'score': hit.get('score', 0.0)
                })

        # Apply BGE Reranker if enabled
        if use_bge_reranker and candidates:
            reranker = get_reranker_model()

            # Build query-document pairs
            pairs = [(query_text, doc['content']) for doc in candidates]
            rerank_scores = reranker.predict(pairs)

            # Add rerank scores and sort
            for i, doc in enumerate(candidates):
                doc['rerank_score'] = rerank_scores[i]

            # Sort by rerank score
            candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

            # Take top_n results
            candidates = candidates[:rerank_top_n]

        # Print results
        # print('TopK results:')
        # for i, doc in enumerate(candidates):
        #     score_type = 'rerank_score' if use_bge_reranker else 'score'
        #     score = doc.get(score_type, 0.0)
        #     print(f"{i+1}. Score: {score:.4f}, Content: {doc['content'][:100]}...")

        return candidates

if __name__ == '__main__':
    ...
    # ret = FullTextRetrieval()
    # ret = HybridTextRetrieval() 
    # ret.query('PointNet')
    # ret.query('Part 2 of EE6221 – Scope')

