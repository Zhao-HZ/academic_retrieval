import os
from typing import List 
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = 'BAAI/bge-m3' 
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

BGE_EMBEDDING_DIM = len(bge_embedding.embed_query("hello"))

def get_bge_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    response = bge_embedding.embed_documents(texts)
    return response
