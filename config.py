import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTOR_DB_DIR = "./chroma_db"
    COLLECTION_NAME = "enterprise_docs"
    DATA_DIR = "./data"
    MODEL_NAME = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"

settings = Config()
