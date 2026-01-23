from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_milvus import Milvus
from pymilvus import \
    (MilvusClient, Collection, DataType, CollectionSchema, Function, FunctionType)
import pdfplumber
from PIL import Image
import os
from typing import List, Tuple, Any
from embedding_model import get_bge_embeddings, BGE_EMBEDDING_DIM
import re
import asyncio
import importlib
from rag_config import *

class IndexConstructionPDF:
    def __init__(self):
        self.client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        self.schema = self.__createSchema()
        self.index_params = self.__create_index_param()
        
        # Create database
        list_db = self.client.list_databases()
        if DB_NAME not in list_db:
            self.client.create_database(db_name=DB_NAME)
        else:
            self.client.use_database(db_name=DB_NAME)

        # Create collection
        list_collection = self.client.list_collections()
        if COLLECTION_NAME not in list_collection:
            self.client.create_collection(collection_name=COLLECTION_NAME, schema=self.schema, index_params=self.index_params)

    def __createSchema(self) -> CollectionSchema:
        analyzer_params = {"tokenizer": "standard", "filter": ['lowercase']}

        schema = MilvusClient.create_schema()
        schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=65535, analyzer_params=analyzer_params, enable_match=True, enable_analyzer=True)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=BGE_EMBEDDING_DIM)
        schema.add_field(field_name='metadata', datatype=DataType.JSON)

        bm25_function = Function(
            name='bm25',
            function_type=FunctionType.BM25,
            input_field_names=['content'],
            output_field_names='sparse_vector'
        )
        schema.add_function(bm25_function)
        
        return schema
   
    def __create_index_param(self):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name='sparse_vector',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='BM25'
        )
        index_params.add_index(field_name='dense_vector', index_type='FLAT', metric_type='IP')
        return index_params
    
    def getImageCaption(self):
        ...

    def segment(self, text: str) -> str:
        wordsegment = importlib.import_module('wordsegment')
        load, segment = wordsegment.load, wordsegment.segment
        load()
        tokens = segment(text)
        return " ".join(tokens)

    # Get the content and images from each page of a PDF file.     
    def analyzePDF(self, pdf_path: str, segment: bool = False, page_range: slice=None) -> Tuple[List[Document], Any]:
        pdf_name = os.path.basename(pdf_path)
        docs, images = [], []

        # prepare the collection for one PDF
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {}
            if page_range:
                pages = pdf.pages[page_range]
            else:
                pages = pdf.pages
            for page_id, page in enumerate(pages, 1):
                text = page.extract_text(layout=True)        
                if not text:
                    text = ''
                metadata['source'] = pdf_name
                metadata['page_number'] = page_id
                metadata['type'] = 'multimodal_page'
                if segment:
                    text = self.segment(text)
                doc = Document(page_content=text, metadata=metadata)
                docs.append(doc)
            for page_id, image in enumerate(pdf.images, 1):
                images.append(image)
        return docs, images

    def splitDocument(self, docs: List[Document], images) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=64,
            separators=['\n\n', '\n', '\n\n\n', '.', ',']
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    # add the chunks of one document(e.g. PDF) into the vector store
    def storeIntoMilvus(self, split_docs: List[Document]) -> VectorStoreRetriever:
        # uuids = [str(uuid4()) for _ in split_docs]
        # self.vector_store.add_documents(split_docs)
        texts = [doc.page_content for doc in split_docs]
        embeddings = get_bge_embeddings(texts)
        entities = []
        for i, doc in enumerate(split_docs):
            entities.append({
                "content": doc.page_content,
                "dense_vector": embeddings[i],
                "metadata": doc.metadata        
            })
        self.client.insert(COLLECTION_NAME, entities)
        print(f"Inserted {len(entities)} documents")

    def processPDF(self, pdf_path, segment: bool=False, page_range: slice=None):
        docs, images = self.analyzePDF(pdf_path, segment, page_range)
        docs = self.splitDocument(docs, images)
        self.storeIntoMilvus(docs)

    def processMultiplePDF(self, pdf_paths: List[str]):
        for path in pdf_paths:
            self.processPDF(path)

    def deleteItemFromDatabase(self, pdf_name):
        # self.vector_store.delete(expr=f'source=="{pdf_name}"')
        ...

    def getIndex(self) -> VectorStoreRetriever:
        # retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        # return retriever
        ...

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
    # construction.processMultiplePDF(pdf_paths, COLLECTION_NAME)
    construction.processPDF(pdf_paths[0], True, slice(11, 17))
    # construction.processPDF(pdf_paths[0], True, None)
      
if __name__ == '__main__':
    asyncio.run(main())

