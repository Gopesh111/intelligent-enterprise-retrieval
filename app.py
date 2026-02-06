from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import get_query_engine

app = FastAPI(title="Intelligent Enterprise Retrieval API")

# Initialize RAG Engine
try:
    query_engine = get_query_engine()
except Exception as e:
    print(f"⚠️ Error loading RAG engine: {e}")
    query_engine = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def health_check():
    return {"status": "online", "system": "Enterprise RAG v1.0"}

@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized. Run ingestion first.")
    
    # Perform RAG Query
    response = query_engine.query(request.query)
    
    # Extract source filenames for transparency
    source_files = list(set([node.metadata['file_name'] for node in response.source_nodes]))

    return {
        "answer": str(response),
        "sources": source_files
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
