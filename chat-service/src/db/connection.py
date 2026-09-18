import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from google import genai

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "multimodal_rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Global client instances
_mongo_client: AsyncIOMotorClient = None
_genai_client: genai.Client = None

def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGO_URI)
    return _mongo_client

def get_db() -> AsyncIOMotorDatabase:
    client = get_mongo_client()
    return client[DB_NAME]

def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    return _genai_client
