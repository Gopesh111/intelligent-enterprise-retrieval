import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import settings

def ingest_data():
    print("🚀 Starting Data Ingestion...")

    # 1. Load Data (Unstructured)
    if not os.path.exists(settings.DATA_DIR):
        os.makedirs(settings.DATA_DIR)
        print(f"⚠️ Created {settings.DATA_DIR}. Please add PDF/TXT files there.")
        return

    documents = SimpleDirectoryReader(settings.DATA_DIR).load_data()
    print(f"📄 Loaded {len(documents)} document(s).")

    # 2. Initialize Vector DB (ChromaDB - Persistent)
    db = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
    chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Create Index & Embeddings
    print("🧠 Generating Embeddings & Storing in Vector DB...")
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("✅ Ingestion Complete! Vector DB is ready.")

if __name__ == "__main__":
    ingest_data()
