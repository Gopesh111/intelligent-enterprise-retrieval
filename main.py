from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from rag_engine import get_query_engine
from ingestion import ingest_data
import shutil
import os
from config import settings
from logging_config import logger

app = FastAPI(title="Enterprise RAG API", version="1.2.0")

# Lazy loading the engine
query_engine = None

@app.on_event("startup")
async def startup_event():
    global query_engine
    try:
        query_engine = get_query_engine()
        logger.info("✅ RAG Engine loaded successfully.")
    except Exception:
        logger.warning("⚠️ No Index found. Please ingest data first.")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_index(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index not ready. Upload documents first.")
    
    response = query_engine.query(request.query)
    
    return {
        "response": str(response),
        "source_nodes": [n.metadata.get('file_name') for n in response.source_nodes]
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a file and triggers re-ingestion."""
    file_path = os.path.join(settings.DATA_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"📥 Received file: {file.filename}")
    
    # Trigger ingestion (In real-world, this would be a background task)
    ingest_data()
    
    # Reload engine to see new data
    global query_engine
    query_engine = get_query_engine()
    
    return {"message": f"File {file.filename} processed and indexed successfully."}
