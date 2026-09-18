import sys
import threading
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import OUTPUT_DIR
from services.job_manager import job_manager
from services.watcher import start_background_watcher
from routes.ingestion_routes import router as ingestion_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion-service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: sweep any stale in-progress jobs from previous server runs
    job_manager.sweep_stale_jobs()
    
    # 2. Launch background directory watcher thread
    watcher_thread = threading.Thread(target=start_background_watcher, daemon=True)
    watcher_thread.start()
    logger.info("[Ingestion Service] Initialized and background directory watcher active.")
    
    yield
    logger.info("[Ingestion Service] Shutting down.")

app = FastAPI(
    title="Enterprise Multimodal RAG Ingestion Service",
    description="Decoupled high-performance document ingestion microservice with Cloudflare R2 and MongoDB Atlas.",
    lifespan=lifespan
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local static output directory for fallback screenshot serving
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")

# Register Modular Ingestion Routes
app.include_router(ingestion_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
