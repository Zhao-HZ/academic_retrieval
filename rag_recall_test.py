"""
RAG召回率测试代码

测试思路:
1. 为每个PDF文档准备相关的测试查询
2. 使用RAG系统检索相关文档
3. 计算Recall@K (K=1,3,5,8)
"""

import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Set, Tuple
from llm.rag.index_optimization import HybridTextRetrieval, FullTextRetrieval
from llm.rag.embedding_model import get_openai_qwen_embeddings
from llm.llm_config import COLLECTION_NAME
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType
from llm.llm_config import MILVUS_URI, MILVUS_TOKEN, DB_NAME

# BGE Reranker
from sentence_transformers import CrossEncoder


TEST_QUERIES = [
    {
        "query": "What are the key differences between PointNet and PointNet++ for 3D point cloud processing?",
        "relevant_docs": ["pointnet.pdf", "pointnet_plusplus.pdf"]
    },
    {
        "query": "Compare PointNet and PointNet++ architectures for 3D deep learning",
        "relevant_docs": ["pointnet.pdf", "pointnet_plusplus.pdf"]
    },
    {
        "query": "How do PointNet and PointNet++ handle permutation invariance in point clouds?",
        "relevant_docs": ["pointnet.pdf", "pointnet_plusplus.pdf"]
    },
    {
        "query": "Hierarchical feature learning in 3D point cloud networks",
        "relevant_docs": ["pointnet.pdf", "pointnet_plusplus.pdf"]
    },

    # ========== DETR ==========
    {
        "query": "How do DN-DETR and Grounding DINO improve transformer-based object detection?",
        "relevant_docs": ["DN-DETR.pdf", "Grounding DINO.pdf"]
    },
    {
        "query": "Compare DN-DETR and Grounding DINO for open-set object detection",
        "relevant_docs": ["DN-DETR.pdf", "Grounding DINO.pdf"]
    },
    {
        "query": "Denoising training and text grounding for DETR-based detectors",
        "relevant_docs": ["DN-DETR.pdf", "Grounding DINO.pdf"]
    },
    {
        "query": "How do transformer DETR models handle object detection?",
        "relevant_docs": ["DN-DETR.pdf", "Grounding DINO.pdf"]
    },

    # ========== TS ==========
    {
        "query": "Compare TimesNet and MOMENT for time series forecasting methods",
        "relevant_docs": ["timesnet.pdf", "MOMENT.pdf"]
    },
    {
        "query": "What are the differences between TimesNet and MOMENT for time series analysis?",
        "relevant_docs": ["timesnet.pdf", "MOMENT.pdf"]
    },
    {
        "query": "Foundation models vs temporal variation modeling in time series",
        "relevant_docs": ["timesnet.pdf", "MOMENT.pdf"]
    },
    {
        "query": "Time series forecasting with deep learning methods",
        "relevant_docs": ["timesnet.pdf", "MOMENT.pdf", "Self-Supervised Learning for Time Series Contrastive or Generative.pdf"]
    },
    {
        "query": "Self-supervised learning for time series representation",
        "relevant_docs": ["MOMENT.pdf", "Self-Supervised Learning for Time Series Contrastive or Generative.pdf"]
    },

    # ========== Transformer Vision ==========
    {
        "query": "How does Swin Transformer compare to DETR-based vision models?",
        "relevant_docs": ["Swin Transformer.pdf", "DN-DETR.pdf"]
    },
    {
        "query": "Hierarchical vision transformers for image classification and detection",
        "relevant_docs": ["Swin Transformer.pdf", "SPP.pdf"]
    },
    {
        "query": "Shifted window mechanism and spatial pyramid pooling in CNNs",
        "relevant_docs": ["Swin Transformer.pdf", "SPP.pdf"]
    },

    # ========== ML Basic ==========
    {
        "query": "Gradient boosting vs deep learning for machine learning tasks",
        "relevant_docs": ["lightgbm.pdf", "MOMENT.pdf"]
    },
    {
        "query": "Compare leaf-wise tree growth with transformer architecture",
        "relevant_docs": ["lightgbm.pdf", "Swin Transformer.pdf"]
    },
    {
        "query": "What are the advantages of gradient boosting and deep learning models?",
        "relevant_docs": ["lightgbm.pdf", "MOMENT.pdf", "Swin Transformer.pdf"]
    },
]


def get_retrieved_docs_by_bm25(query: str, top_k: int = 10) -> Set[str]:
   
    client = MilvusClient(url=MILVUS_URI, token=MILVUS_TOKEN)
    client.use_database(DB_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query],
        anns_field='sparse_vector',
        limit=top_k,
        output_fields=['metadata']
    )

    retrieved = set()
    for result in results[0]:
        metadata = result['entity'].get('metadata', {})
        source = metadata.get('source', '')
        if source:
            retrieved.add(source)

    return retrieved


def get_retrieved_docs_by_dense(query: str, top_k: int = 10) -> Set[str]:
    client = MilvusClient(url=MILVUS_URI, token=MILVUS_TOKEN)
    client.use_database(DB_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    query_vector = get_openai_qwen_embeddings([query])[0]

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field='dense_vector',
        limit=top_k,
        output_fields=['metadata']
    )

    retrieved = set()
    for result in results[0]:
        metadata = result['entity'].get('metadata', {})
        source = metadata.get('source', '')
        if source:
            retrieved.add(source)

    return retrieved


def get_retrieved_docs_by_hybrid(query: str, top_k: int = 10) -> Set[str]:
    
    client = MilvusClient(url=MILVUS_URI, token=MILVUS_TOKEN)
    client.use_database(DB_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    query_vector = get_openai_qwen_embeddings([query])[0]
    search_param_1 = {
        'data': [query_vector],
        'anns_field': "dense_vector",
        'param': {'nprobe': 10},
        'limit': top_k
    }
    request1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        'data': [query],
        'anns_field': "sparse_vector",
        'param': {'nprobe': 10},
        'limit': top_k
    }
    request2 = AnnSearchRequest(**search_param_2)

    rrf_ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={
            'reranker': 'rrf',
            "k": 100
        }
    )

    reqs = [request1, request2]

    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=reqs,
        ranker=rrf_ranker,
        limit=top_k,
        output_fields=['metadata']
    )

    retrieved = set()
    for hits in res:
        for hit in hits:
            metadata = hit['entity'].get('metadata', {})
            source = metadata.get('source', '')
            if source:
                retrieved.add(source)

    return retrieved


_reranker_model = None

def get_reranker_model(model_name: str = "BAAI/bge-reranker-large") -> CrossEncoder:
    
    global _reranker_model
    if _reranker_model is None:
        print(f"Load Reranker Model: {model_name}")
        _reranker_model = CrossEncoder(model_name, max_length=512)
        print("Reranker Loaded")
    return _reranker_model


def get_retrieved_docs_with_rerank(
    query: str,
    top_k: int = 10,
    retrieval_method: str = "hybrid",
    rerank_top_n: int = 5,
    use_rerank: bool = True
) -> Set[str]:
    client = MilvusClient(url=MILVUS_URI, token=MILVUS_TOKEN)
    client.use_database(DB_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    candidate_top_k = max(top_k * 3, 20)  

    if retrieval_method == "bm25":
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query],
            anns_field='sparse_vector',
            limit=candidate_top_k,
            output_fields=['metadata', 'content']
        )
    elif retrieval_method == "dense":
        query_vector = get_openai_qwen_embeddings([query])[0]
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            anns_field='dense_vector',
            limit=candidate_top_k,
            output_fields=['metadata', 'content']
        )
    else:  # hybrid
        query_vector = get_openai_qwen_embeddings([query])[0]
        search_param_1 = {
            'data': [query_vector],
            'anns_field': "dense_vector",
            'param': {'nprobe': 10},
            'limit': candidate_top_k
        }
        request1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            'data': [query],
            'anns_field': "sparse_vector",
            'param': {'nprobe': 10},
            'limit': candidate_top_k
        }
        request2 = AnnSearchRequest(**search_param_2)

        rrf_ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                'reranker': 'rrf',
                "k": 100
            }
        )

        reqs = [request1, request2]
        results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=reqs,
            ranker=rrf_ranker,
            limit=candidate_top_k,
            output_fields=['metadata', 'content']
        )

    candidates = []
    for result in results[0]:
        metadata = result['entity'].get('metadata', {})
        text = result['entity'].get('content', '')
        source = metadata.get('source', '')
        if source and text:
            candidates.append({
                'source': source,
                'content': text[:1000], 
                'score': result.get('score', 0.0)
            })

    if not use_rerank:
        retrieved = set()
        for doc in candidates[:top_k]:
            retrieved.add(doc['source'])
        return retrieved

    reranker = get_reranker_model()

    pairs = [(query, doc['content']) for doc in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, doc in enumerate(candidates):
        doc['rerank_score'] = rerank_scores[i]

    
    reranked_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    retrieved = set()
    for doc in reranked_candidates[:rerank_top_n]:
        retrieved.add(doc['source'])

    return retrieved


get_retrieved_docs = get_retrieved_docs_by_hybrid


def calculate_recall(retrieved: Set[str], relevant: Set[str]) -> Tuple[int, int, float]:
    relevant_retrieved = retrieved & relevant
    hit_count = len(relevant_retrieved)
    total_relevant = len(relevant)
    recall = hit_count / total_relevant if total_relevant > 0 else 0.0
    return hit_count, total_relevant, recall


def calculate_precision(retrieved: Set[str], relevant: Set[str]) -> Tuple[int, int, float]:
    
    relevant_retrieved = retrieved & relevant
    hit_count = len(relevant_retrieved)
    total_retrieved = len(retrieved)
    precision = hit_count / total_retrieved if total_retrieved > 0 else 0.0
    return hit_count, total_retrieved, precision


def evaluate_rag_system(test_queries: List[Dict], top_k: int = 5, retrieval_method: str = "hybrid") -> Dict:
    
    if retrieval_method == "bm25":
        retrieve_func = get_retrieved_docs_by_bm25
    elif retrieval_method == "dense":
        retrieve_func = get_retrieved_docs_by_dense
    elif retrieval_method == "hybrid+rerank":
        def wrapper(query, top_k):
            return get_retrieved_docs_with_rerank(query, top_k, retrieval_method="hybrid",
                                                   rerank_top_n=top_k, use_rerank=True)
        retrieve_func = wrapper
    else:
        retrieve_func = get_retrieved_docs_by_hybrid

    results = {
        "total_queries": len(test_queries),
        "retrieval_method": retrieval_method,
        "recall_at_k": {},
        "precision_at_k": {},
        "detailed_results": []
    }

    k_values = [5, 8, 10, 15, 20]
    recall_sum = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}

    for i, test_item in enumerate(test_queries):
        query = test_item["query"]
        relevant_docs = set(test_item["relevant_docs"])

        retrieved_docs = retrieve_func(query, top_k=max(k_values))

        query_result = {
            "query": query,
            "relevant_docs": list(relevant_docs),
            "retrieved_docs": list(retrieved_docs),
            "recalls": {},
            "precisions": {}
        }

        for k in k_values:
            retrieved_k = set(list(retrieved_docs)[:k]) if len(retrieved_docs) >= k else retrieved_docs

            hit, total, recall = calculate_recall(retrieved_k, relevant_docs)
            hit, total_ret, precision = calculate_precision(retrieved_k, relevant_docs)

            recall_sum[k] += recall
            precision_sum[k] += precision

            query_result["recalls"][k] = recall
            query_result["precisions"][k] = precision

        results["detailed_results"].append(query_result)

        print(f"Processed query {i+1}/{len(test_queries)}: {query[:50]}...")

    num_queries = len(test_queries)
    for k in k_values:
        results["recall_at_k"][k] = recall_sum[k] / num_queries
        results["precision_at_k"][k] = precision_sum[k] / num_queries

    return results


def print_evaluation_results(results: Dict):

    for k, recall in results["recall_at_k"].items():
        print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

    for k, precision in results["precision_at_k"].items():
        print(f"  Precision@{k}: {precision:.4f} ({precision*100:.2f}%)")

    for i, detail in enumerate(results["detailed_results"][:5]):
        print(f"\nQuery:  {i+1}: {detail['query']}")
        print(f"  Expected Doc: {detail['relevant_docs']}")
        print(f"  Retrieved Doc: {detail['retrieved_docs']}")
        print(f"  Recall@5: {detail['recalls'][5]:.4f}")


def test_single_query(query: str, relevant_docs: List[str], top_k: int = 5, retrieval_method: str = "hybrid"):
    print(f"\nQuery: {query}")
    print(f"Expected Recall: {relevant_docs}")

    if retrieval_method == "bm25":
        retrieved = get_retrieved_docs_by_bm25(query, top_k)
    elif retrieval_method == "dense":
        retrieved = get_retrieved_docs_by_dense(query, top_k)
    else:
        retrieved = get_retrieved_docs_by_hybrid(query, top_k)

    print(f"Actual Recall: {list(retrieved)}")

    relevant_set = set(relevant_docs)
    hit, total, recall = calculate_recall(retrieved, relevant_set)
    print(f"Hit: {hit}/{total}, Recall@{top_k}: {recall:.4f}")

    return retrieved


def main():
    

    # methods = ["bm25", "dense", "hybrid", "hybrid+rerank"]
    methods = ["bm25", "dense", "hybrid"]
    # methods = ["hybrid+rerank"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Test Method: {method}")
        print("="*60)

        results = evaluate_rag_system(TEST_QUERIES, top_k=15, retrieval_method=method)
        print_evaluation_results(results)

    # 测试单个查询示例
    # print("\n" + "="*60)
    # print("单个查询测试示例 (hybrid)")
    # print("="*60)
    # test_single_query(
    #     "What is PointNet for 3D point cloud processing?",
    #     ["pointnet.pdf"],
    #     top_k=5,
    #     retrieval_method="hybrid"
    # )

    return results


if __name__ == "__main__":
    main()
