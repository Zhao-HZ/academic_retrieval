import pdfplumber as pdf
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from llm.llm_config import load_openai_qwen

def get_react_prompt() -> str:
    """Custom ReAct prompt for the academic retrieval agent"""
    return """You are an academic assistant agent with access to various tools.

Your capabilities:
1. Process and analyze PDF documents
2. Retrieve information from a knowledge base (RAG)
3. Save PDF user designated to the RAG vector database
4. Perform calculations
5. Save notes
6. 

Available tools:
- doc_augumented: Augment document using RAG model (input: doc_path)
- rag_retrieval: Retrieve documents using RAG model (input: query_text, top_k)
- db_insert_retrieval_history: Insert retrieval history (input: field, question, answer)
- calculator: Calculate mathematical expressions (input: expression)
- PDF_process: Extract text from PDF file (input: pdf_path)
- save_note_to_md: Save note to Markdown file (input: note)
- pdf_rag: Process PDF using RAG model (input: pdf_path)
- markdown_rag_by_path: Process Markdown using RAG model (input: mddown_path)
- get_current_time: Get current time (no input)

Instructions:
- Use the tools to help answer user questions
- If the user asks about PDF content, use PDF_process to read the PDF first
- If the user asks to retrieve information from the knowledge base, use rag_retrieval
- If the user asks for an outline of a PDF or to briefly explain the content of the PDF, use PDF_process to read it and provide an outline
or the explaination.
- Always provide clear and helpful answers based on the tool results

Remember: You must use the tools available to you. Format your response with the appropriate tool calls."""

load_model = load_openai_qwen

def create_agent__():
    """Create the ReAct agent with tools"""
    from llm.tools import (
        doc_augumented,
        rag_retrieval,
        db_insert_retrieval_history,
        calculator,
        PDF_process,
        save_note_to_md,
        pdf_rag,
        markdown_rag_by_path,
        get_current_time
    )
    from llm.llm_config import load_model

    # Load the LLM
    llm = load_model()

    # Get tools
    tools = [
        doc_augumented,
        rag_retrieval,
        db_insert_retrieval_history,
        calculator,
        PDF_process,
        save_note_to_md,
        pdf_rag,
        markdown_rag_by_path,
        get_current_time
    ]

    # Create the agent using the new LangChain API with memory
    memory = MemorySaver()
    system_prompt = get_react_prompt()
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )

    return agent



# For testing when running directly
if __name__ == "__main__":
    ...
    # test()