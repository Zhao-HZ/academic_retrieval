You task is to develop an agent for personal knowledge base. You should implement it on @./llm/workflow.py file.
1. The agent is of ReAct style.
2. The agent should support short context memory, the context is stored on a single chat.   
3. Design a prompt of the ReAct agent by your own. 
4. The tools that this agent can use are defined on @.llm/tools.py.
5. Users may ask the agent to generate an outline of a PDF file or retrieval related information from the Milvus database. You should try to extend the functionality. However, you should not extend it beyond the tools. 
6. Finally, implement a function called test on @./llm/workflow.py to test the result. This function supports multiple-turn agent conversation.

The langchain skill I defined might be used to implement the agent.
