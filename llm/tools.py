import math
import os
from langchain.tools import tool
from llm.rag.index_optimization import HybridTextRetrieval
from llm.rag.index_construction import IndexConstruction
from typing import Literal 

@tool("doc_augumented")
def doc_augumented(doc_path: str) -> bool:
    """Augument document using RAG model"""
    index_construction = IndexConstruction()
    index_construction.processPDF(doc_path, segment=True)
    return True

@tool("rag_retrieval")
def rag_retrieval(query_text: str, top_k: int = 5) -> list:
    """Retrieve documents using RAG model"""
    retriever = HybridTextRetrieval()
    return retriever.query(query_text, use_bge_reranker=False, top_k=top_k)

@tool("db_insert_retrieval_history")
def db_insert_retrieval_history(field:str, question:str, answer:str) -> bool:
    """Insert retrieval history into database"""
    from db.db_manager import DBManager
    db_manager = DBManager()
    db_manager.insert_retrieval_history(field, question, answer)

    return True

@tool("calculator")
def calculator(expression: str) -> float:
    """Calculate the expression"""
    
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "log": math.log,
        "ln": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "e": math.e,
        "pi": math.pi,
    }
    try:
        return eval(expression, globals(), allowed_names)
    except Exception as e:
        raise ValueError(f"Error: {e}")

@tool('PDF_process')
def PDF_process(pdf_path: str) -> str:
    """Process PDF file and return its content for LLM use."""
    import pdfplumber as pdf
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_name = os.path.basename(pdf_path)
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text(layout=True)
            if text:
                full_text.append(f"--- Page {page_num} ---\n{text}")
            else:
                full_text.append(f"--- Page {page_num} ---\n[No text content]")

    return f"Source: {pdf_name}\n\n" + "\n\n".join(full_text)

@tool('save note')
def save_note_to_md(note: str) -> bool:
    """Save note to Markdown file"""
    with open("note.md", "w") as f:
        f.write(note)
    return True

@tool('pdf rag')
def pdf_rag(pdf_path: str) -> bool:
    """Process PDF file using RAG model"""
    construction = IndexConstruction()    
    construction.processPDF(pdf_path)

@tool("markdown rag by path")
def markdown_rag_by_path(mddown_path: str) -> bool:
    """Process Markdown file using RAG model"""
    construction = IndexConstruction()    
    construction.processMarkdown(mddown_path)
    return True

@tool('get current time')    
def get_current_time() -> str:
    """Get current time"""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
