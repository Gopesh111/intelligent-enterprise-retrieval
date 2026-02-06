import chromadb
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import settings

# Custom Prompt Template to make answers professional
QA_PROMPT_TMPL = (
    "You are an intelligent enterprise assistant. Use the context below to answer the query.\n"
    "If the answer isn't in the context, say 'I cannot find that information in the internal documents.'\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

def get_query_engine():
    # Load DB
    db = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
    chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Set custom prompt
    qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

    # Return engine with streaming enabled for lower perceived latency
    return index.as_query_engine(
        similarity_top_k=3, 
        text_qa_template=qa_prompt,
        streaming=False # Keep false for simple API response, True for advanced websockets
    )
