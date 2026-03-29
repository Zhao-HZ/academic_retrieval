import asyncio
import os
from typing import List
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from llm.llm_config import OPENAI_QWEN_BASE_URL_SG

HF = 0
OLLAMA = 0
OPENAI = 1
QWEN = 1

if HF:
    # BGE embeddings
    model_name = 'BAAI/bge-m3'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {"normalize_embeddings": True}
    hf_bge_embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    HF_BGE_EMBEDDING_DIM = len(hf_bge_embedding.embed_query("hello"))
    def get_hf_bge_embeddings(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = hf_bge_embedding.embed_documents(texts)
        return response

# Ollama Qwen embeddings
if OLLAMA:
    ollama_qwen_embedding = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )
    OLLAMA_QWEN_EMBEDDING_DIM = len(ollama_qwen_embedding.embed_query("hello"))

    def get_ollama_qwen_embeddings(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = ollama_qwen_embedding.embed_documents(texts)
        return response

if OPENAI:
    OPENAI_QWEN_EMBEDDING_DIM = 1024
    # For Singapore region
    client = OpenAI(
        api_key="sk-e82ac726670c4c36838397fbe801fc22",
        base_url=OPENAI_QWEN_BASE_URL_SG
    )
    def get_openai_qwen_embeddings(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Process in batches of 10 (API limit)
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            completion = client.embeddings.create(
                model="text-embedding-v3",
                input=batch,
                dimensions=OPENAI_QWEN_EMBEDDING_DIM,
            )
            all_embeddings.extend([e.embedding for e in completion.data])

        return all_embeddings

if QWEN:
    def get_qwen_embeddings_model():
        embeddings_model = OpenAIEmbeddings(
            api_key="sk-e82ac726670c4c36838397fbe801fc22",
            base_url=OPENAI_QWEN_BASE_URL_SG,
            model="text-embedding-v3",
            check_embedding_ctx_length=False,
            chunk_size=10  # API limit: batch size should not be larger than 10
        )
        return embeddings_model

    def get_qwen_embeddings(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings_model = get_qwen_embeddings_model()
        response = embeddings_model.embed_documents(texts)
        return response
 
def test():
    texts = ["hello", "world"]
    response = get_qwen_embeddings(texts)
    print(response)

# test() 