import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import settings

def get_query_engine():
    # 1. Connect to existing Vector DB
    db = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
    chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 2. Load Index from Vector Store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

    # 3. Create Query Engine (The RAG Logic)
    return index.as_query_engine(similarity_top_k=3)
