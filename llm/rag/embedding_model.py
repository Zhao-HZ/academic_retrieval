import os
from typing import List 
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

HF = 0
OLLAMA = 1


model_name = 'BAAI/bge-m3' 
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

if HF:
    hf_bge_embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    BGE_EMBEDDING_DIM = len(hf_bge_embedding.embed_query("hello"))
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
    QWEN_EMBEDDING_DIM = len(ollama_qwen_embedding.embed_query("hello"))

    def get_ollama_qwen_embeddings(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = ollama_qwen_embedding.embed_documents(texts)
        return response
