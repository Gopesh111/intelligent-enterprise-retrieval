import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import settings
from logging_config import logger

def ingest_data():
    logger.info("🚀 Starting Ingestion Pipeline...")

    # 1. Load Data
    if not os.path.exists(settings.DATA_DIR):
        os.makedirs(settings.DATA_DIR)
        logger.warning(f"Directory {settings.DATA_DIR} created. Add PDFs to run ingestion.")
        return

    documents = SimpleDirectoryReader(settings.DATA_DIR).load_data()
    logger.info(f"📄 Loaded {len(documents)} raw documents.")

    # 2. Advanced Chunking (Critical for RAG accuracy)
    # Splitting text into 1024-token chunks with overlap to maintain context
    transformations = [SentenceSplitter(chunk_size=1024, chunk_overlap=200)]

    # 3. Vector Database Setup
    db = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
    chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Indexing
    logger.info("🧠 Generating Embeddings...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=transformations,
        show_progress=True
    )
    logger.info("✅ Ingestion Complete. Vector Store persisted.")

if __name__ == "__main__":
    ingest_data()
