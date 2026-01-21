from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import MilvusClient, Collection, DataType
import pdfplumber
from PIL import Image
import os
from typing import List, Tuple, Any
from embedding_model import bge_embedding
import re
import asyncio
from uuid import uuid4

DB_NAME = "academic_retrieval_db"
COLLECTION_NAME = "academic_retrieval_collection"
MILVUS_URI = 'http://localhost:19530'

class IndexConstructionPDF:
    def __init__(self):
        self.client = MilvusClient(
            uri=MILVUS_URI,
            token='root:Milvus'
        )
        # Create database
        list_db = self.client.list_databases()
        if DB_NAME not in list_db:
            self.client.create_database(db_name=DB_NAME)
        else:
            self.client.use_database(db_name=DB_NAME)

        # Create collection
        list_collection = self.client.list_collections()
        if COLLECTION_NAME not in list_collection:
            self.client.create_collection(collection_name=COLLECTION_NAME)

        # create vector store
        self.vector_store = Milvus(
            embedding_function=bge_embedding, collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI, "token": "root:Milvus", "db_name": DB_NAME},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            consistency_level="Strong",
        )
        self.all_docs = []
        
    def getImageCaption(self):
        ...

    # Get the content and images from each page of a PDF file.     
    def analyzePDF(self, pdf_path: str) -> Tuple[List[Document], Any]:
        pdf_name = os.path.basename(pdf_path)
        docs, images = [], []
        # prepare the collection for one PDF
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {}
            for page_id, page in enumerate(pdf.pages, 1):
                text = page.extract_text()        
                if not text:
                    text = ''
                metadata['source'] = pdf_name
                metadata['page_number'] = page_id
                metadata['type'] = 'multimodal_page'
                doc = Document(page_content=text, metadata=metadata)

                docs.append(doc)

            for page_id, image in enumerate(pdf.images, 1):
                images.append(image)
        return docs, images

    def splitDocument(self, docs: List[Document], images) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=['\n\n', '\n', '\n\n\n', '.', ',']
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    # add the chunks of one document(e.g. PDF) into the vector store
    def storeIntoMilvus(self, split_docs: List[Document]) -> VectorStoreRetriever:
        uuids = [str(uuid4()) for _ in split_docs]
        self.vector_store.add_documents(split_docs)

    def processPDF(self, pdf_path):
        docs, images = self.analyzePDF(pdf_path)
        docs = self.splitDocument(docs, images)
        self.storeIntoMilvus(docs)

    def processMultiplePDF(self, pdf_paths: List[str], collection_names):
        for collection_name, path in zip(collection_names, pdf_paths):
            self.processPDF(path)

    def deleteItemFromDatabase(self, pdf_name):
        self.vector_store.delete(expr=f'source=="{pdf_name}"')

    def getIndex(self) -> VectorStoreRetriever:
        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        return retriever

    # def test(self):
    #     docs, images = self.processPDF()
    #     self.splitDocument(docs, images)

pdf_paths = [
    './llm/rag/res/Review_Recent_Advances_in_Sensor_Fusion_Monitoring_and_Control_Strategies_in_Laser_Powder_Bed_Fusion_A_Review.pdf',
]

collection_names = [
    'review'
]

async def main():
    construction = IndexConstructionPDF()
    construction.processMultiplePDF(pdf_paths, collection_names)

if __name__ == '__main__':
    asyncio.run(main())
