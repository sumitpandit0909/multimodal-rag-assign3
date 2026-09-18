import os
import sys
from pathlib import Path
from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env configurations
load_dotenv()
load_dotenv(SRC_DIR.parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.chat_routes import router as chat_router

app = FastAPI(
    title="Enterprise Multimodal RAG Chat Service",
    description="Decoupled Agentic RAG chat microservice powered by LangGraph, MongoDB Atlas, and Gemma-3-27b-it",
    version="2.0.0"
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve generated page screenshots
PROCESSED_OUTPUT = SRC_DIR.parent.parent / "processed_output"
PROCESSED_OUTPUT.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(PROCESSED_OUTPUT)), name="static")

# Register modular API routes
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)