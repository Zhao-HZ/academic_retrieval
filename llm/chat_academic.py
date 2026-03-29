from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import rag_retrieval
from llm.llm_config import OPENAI_QWEN_BASE_URL_SG
from langchain.messages import HumanMessage

# chat_model = ChatOpenAI(
#     base_url=OPENAI_QWEN_BASE_URL_SG,
#     api_key="sk-e82ac726670c4c36838397fbe801fc22",
#     model="qwen-plus",
# )

# agent = create_agent(
#     model=chat_model,
#     tools=[rag_retrieval],
#     system_prompt="你需要根据用户的问题，从数据库中检索相关的文档, 使用rag_retrieval工具提取相关信息并整合, 如果工具调用失败,直接提示用户,不用自己回答",
#     # verbose=True
# ) 


# result = agent.invoke(
#     {"messages": [HumanMessage(content="What is PointNet?")]}
# )
# print(result)