import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Ensure 'src' is in python path for local imports (db, agent)
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env from root and current directory
load_dotenv()
load_dotenv(SRC_DIR.parent.parent / ".env")

import shutil
from fastapi import FastAPI, status, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai

from db.memory import MongoChatMemory
from agent.tools import search_vector_store
from agent.agent_graph import run_agentic_rag

# Try importing ingestion worker logic for direct pipeline triggering
INGESTION_SRC = SRC_DIR.parent.parent / "ingestion-services" / "src"
if INGESTION_SRC.exists() and str(INGESTION_SRC) not in sys.path:
    sys.path.insert(0, str(INGESTION_SRC))

try:
    from main import process_file
except Exception:
    process_file = None



from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Enterprise Multimodal RAG Chat Service")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED_OUTPUT = SRC_DIR.parent.parent / "processed_output"
PROCESSED_OUTPUT.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(PROCESSED_OUTPUT)), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str

class SourceNode(BaseModel):
    citation_id: int
    file_name: str
    source_type: str
    screenshot_url: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    raw_data: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceNode]

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "multimodal_rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    try:
        # Check Atlas connectivity
        await client.admin.command("ping")
        # Check if vector chunks exist
        vector_count = await db["vectors"].count_documents({})
        return {
            "status": "healthy",
            "database_connected": True,
            "vectors_ingested_count": vector_count,
            "ready_for_queries": vector_count > 0
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database_connected": False,
            "error": str(e)
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    memory = MongoChatMemory(db)
    history = await memory.get_history(request.session_id)
    
    # 1. Retrieve relevant vectors
    sources = await search_vector_store(request.message, db, genai_client)
    
    # 2. Synthesize answer with citations
    rag_result = await run_agentic_rag(request.message, history, sources, genai_client)
    
    # 3. Store conversation turn
    await memory.add_turn(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=rag_result["answer"],
        sources=rag_result["source_nodes"]
    )
    
    return ChatResponse(
        session_id=request.session_id,
        answer=rag_result["answer"],
        sources=rag_result["source_nodes"]
    )

DATA_DIR = SRC_DIR.parent.parent / "data_drop"
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if process_file:
        background_tasks.add_task(process_file, file_path, genai_client)
        return {
            "status": "processing",
            "message": f"'{file.filename}' uploaded. Ingestion pipeline is running.",
            "file_name": file.filename
        }
    else:
        return {
            "status": "queued",
            "message": f"'{file.filename}' saved to data_drop.",
            "file_name": file.filename
        }

@app.get("/documents")
async def get_documents():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$file_name",
                    "source_type": {"$first": "$source_type"},
                    "chunks": {"$sum": 1},
                    "screenshot_url": {"$first": "$screenshot_url"}
                }
            },
            {
                "$project": {
                    "file_name": "$_id",
                    "source_type": 1,
                    "chunks": 1,
                    "screenshot_url": 1,
                    "_id": 0
                }
            },
            {"$sort": {"file_name": 1}}
        ]
        cursor = db["vectors"].aggregate(pipeline)
        docs = await cursor.to_list(length=100)
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        return {"documents": [], "error": str(e)}