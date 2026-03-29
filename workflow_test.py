from llm.react_agent import chain

# result1 = chain.invoke(
#     {"prompt": "By using knowledge_retrieval tool, answer the following question. return error if you fail to retrieve information from tool: What is HMM?"}
# )    

result2 = chain.invoke(
    {"prompt": "Generate a mindmap about NLP"}
)

print(result2)