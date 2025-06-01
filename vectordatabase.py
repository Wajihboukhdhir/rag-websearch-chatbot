# vectordatabase.py
import os
import time
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def store_documents(db_path, collection, docs, chunk_size, chunk_overlap, batch_limit, delay):
    """Handles document storage in Chroma vector database"""
    try:
        embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as model_error:
        print(f"Embedding model error: {model_error}")
        raise

    try:
        db_instance = Chroma(
            collection_name=collection,
            embedding_function=embedding_model,
            persist_directory=db_path
        )
    except Exception as db_error:
        print(f"Database initialization error: {db_error}")
        raise

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        document_chunks = splitter.split_documents(docs)
    except Exception as split_error:
        print(f"Document splitting error: {split_error}")
        raise

    for start_idx in range(0, len(document_chunks), batch_limit):
        end_idx = start_idx + batch_limit
        current_batch = document_chunks[start_idx:end_idx]
        db_instance.add_documents(current_batch)
        time.sleep(delay)
        
    db_instance.persist()