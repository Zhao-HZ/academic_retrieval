DB_NAME = "academic_retrieval_db"
COLLECTION_NAME = "academic_retrieval_collection"
MILVUS_URI = 'http://localhost:19530'
MILVUS_TOKEN = 'root:Milvus'

DEEPSEEK = 0
OPENAI_QWEN_SG = 1
OPENAI_QWEN_BJ = 0
OPENAI_QWEN_BASE_URL_SG="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
OPENAI_QWEN_BASE_URL_BJ="https://dashscope.aliyuncs.com/compatible-mode/v1"

def get_milvus_client():
    from pymilvus import MilvusClient
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    return client

from langchain_openai import ChatOpenAI

def load_openai_qwen():
    from llm.llm_config import OPENAI_QWEN_BASE_URL_SG
    url = OPENAI_QWEN_BASE_URL_SG if OPENAI_QWEN_SG else OPENAI_QWEN_BASE_URL_BJ
    chat_model = ChatOpenAI(
        base_url=url,
        api_key="sk-e82ac726670c4c36838397fbe801fc22",
        model="qwen-plus",
    )
    return chat_model

def load_model():
    if OPENAI_QWEN_SG or OPENAI_QWEN_BJ:
        return load_openai_qwen()
    
        
