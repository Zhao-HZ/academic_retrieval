from langchain.agents import create_agent
from llm.tools import *
from llm.llm_config import load_openai_qwen
from typing import Literal

class Agents():
    def __init__(self, models:Literal['openai_qwen', 'deepseek', 'ollama_deepseek']='openai_qwen'):
        self.chat_model = None
        if models == 'openai_qwen':
            self.chat_model = load_openai_qwen()
        elif models == 'deepseek':
            ...
        
    def router_agent(self):
        PROMPT = """
        
        """
        agent = create_agent(
            system_prompt=PROMPT,
            model=self.chat_model,
            tools=[rag_retrieval],
        )
        return agent
        