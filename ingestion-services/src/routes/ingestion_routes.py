import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from pymongo import MongoClient

from core.config import MONGO_URI, MONGO_DB, DATA_DIR, genai_client
from services.job_manager import job_manager
from services.pipeline_orchestrator import process_file
from storage.r2_storage import r2_storage
from models.schemas import UploadResponse, JobStatus, DocumentListResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])

@router.get("/health", response_model=HealthResponse)
def health():
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        m_client.admin.command("ping")
        coll = m_client[MONGO_DB]["vectors"]
        count = coll.count_documents({})
        m_client.close()
        return HealthResponse(
            status="healthy",
            service="ingestion-service",
            database_connected=True,
            vectors_ingested_count=count,
            r2_storage_configured=r2_storage.is_configured()
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            service="ingestion-service",
            database_connected=False,
            error=str(e),
            r2_storage_configured=r2_storage.is_configured()
        )

@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_manager.init_job(job_id, file.filename)
    
    file_path = DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        job_manager.fail_job(job_id, "upload", f"Failed to save uploaded file: {str(e)}", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Run tracked pipeline in background
    background_tasks.add_task(process_file, file_path, genai_client, job_id)
    return UploadResponse(
        job_id=job_id,
        file_name=file.filename,
        status="processing",
        message=f"'{file.filename}' uploaded successfully. Tracking pipeline execution."
    )

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@router.get("/jobs")
def list_jobs():
    return {"jobs": job_manager.list_jobs(limit=25)}

@router.get("/documents")
def get_documents():
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        coll = m_client[MONGO_DB]["vectors"]
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
        docs = list(coll.aggregate(pipeline))
        m_client.close()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        return {"documents": [], "error": str(e)}
