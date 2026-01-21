import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = 'BAAI/bge-m3' 
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
