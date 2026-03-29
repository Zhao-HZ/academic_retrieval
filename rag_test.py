from llm.rag.index_optimization import HybridTextRetrieval, FullTextRetrieval
from llm.rag.index_construction import IndexConstruction
from llm.llm_config import OPENAI_QWEN_BASE_URL_SG
from llm.llm_config import load_openai_qwen
from llm.llm_config import get_milvus_client
from llm.tools import rag_retrieval
# from llm.rag.embedding_model import get_openai_qwen_embeddings
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from glob import glob
from pymilvus import MilvusClient
from time import time

def time_record(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} time: {end - start:.2f} seconds")
        return result
    return wrapper

@time_record
def test_index_construction():
    index_construction = IndexConstruction()
    pdf_paths = glob('./res/pdf/*.pdf')
    index_construction.processMultiplePDF(pdf_paths)
   
def test_index_optimization():
    hybrid_retrieval = HybridTextRetrieval()
    query = "PointNet"
    hybrid_retrieval.query(query)
   
@time_record
def test_chat():
    chat_model = load_openai_qwen()
    agent = create_agent(
        model=chat_model,
        tools=[rag_retrieval],
        system_prompt="你需要根据用户的问题，必须从数据库中检索相关的文档进行回答, \
            使用rag_retrieval工具提取相关信息并整合, 如果工具调用失败或者检索不到相关信息,直接返回\"Information not found.\"或者\"找不到相关信息.\"",
    ) 

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is PyTorch?")]}
    )
    print(result)
    # return result 

def drop_collection():
    # client = MilvusClient(
    #     uri="http://localhost:19530",
    #     token="root:Milvus"
    # )
    client = get_milvus_client()
    client.use_database('academic_retrieval_db')
    client.drop_collection(collection_name=client.list_collections()[0])
    # print(client.list_collections())

def list_collections():
    client = get_milvus_client() 
    client.use_database('academic_retrieval_db')
    print(client.list_collections())

def list_chunk():
    ...



# test_index_construction()
# test_chat() 
# drop_collection()
list_collections()
