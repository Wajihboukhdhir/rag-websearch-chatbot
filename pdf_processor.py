#pdf_processor.py
import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import config
from vectordatabase import store_documents

def get_pdf_content(file_path):
    """Extracts and combines text content from PDF pages"""
    pdf_loader = PyPDFLoader(file_path)
    document_pages = pdf_loader.load_and_split()
    logging.debug(f"Extracted {len(document_pages)} pages from {file_path}")
    return "\n\n".join([page.page_content.strip() for page in document_pages])

def handle_pdf_processing(input_dir, output_dir, collection_name, config_settings):
    """Processes PDF files and stores them in vector database"""
    processed_docs = []
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):
            continue
            
        full_path = os.path.join(input_dir, filename)
        try:
            extracted_text = get_pdf_content(full_path)
            doc = Document(
                page_content=extracted_text,
                metadata={"source": full_path}
            )
            processed_docs.append(doc)
            logging.info(f"Processed {full_path}")
        except Exception as error:
            logging.error(f"Failed processing {filename}: {error}")
            raise

    store_documents(
        db_path=output_dir,
        collection=collection_name,
        docs=processed_docs,
        chunk_size=config_settings['CHUNK_SIZE'],
        chunk_overlap=config_settings['CHUNK_OVERLAP'],
        batch_limit=config_settings['BATCH_SIZE'],
        delay=config_settings['SLEEP_SECONDS']
    )
    logging.info(f"Stored {len(processed_docs)} documents in {collection_name}")

if __name__ == "__main__":
    load_dotenv()
    config_settings = config.read_config()  # Get the configuration dictionary
    
    print(f"Processing documents for {config_settings['topic_name']}")
    print(f"Source directory: {config_settings['pdf_base_dir']}")
    print(f"Database location: {config_settings['chroma_base_dir']}")
    
    logging.basicConfig(level=config_settings['LOG_LEVEL'])
    logger = logging.getLogger(__name__)

    pdf_directory = "C:/Users/Wajih/Desktop/Projects Wajih/Graduation/pdf"
    db_directory = os.path.join(config_settings['chroma_base_dir'], config_settings['chroma_collection'])
    handle_pdf_processing(pdf_directory, db_directory, config_settings['chroma_collection'], config_settings)