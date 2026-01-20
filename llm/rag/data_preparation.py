import pymilvus
import pdfplumber
from PIL import Image

class DataPreparationPDF:
    def __init__(self):
        ...

    def getImageCaption(self):
        

    def processPDF(self, pdf_path):
        docs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_id, page in enumerate(pdf.pages, 1):
                text = page.extract_text()        
                print(text)

                for img in page.images:
                    

        
if __name__ == '__main__':
    preparation = DataPreparationPDF()
    preparation.processPDF('./llm/rag/res/Review - Recent Advances in Sensor Fusion Monitoring and Control Strategies in Laser Powder Bed Fusion A Review.pdf')