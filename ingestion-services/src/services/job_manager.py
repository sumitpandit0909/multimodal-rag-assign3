import time
import logging
import threading
import traceback
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from core.config import MONGO_URI, MONGO_DB

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages the lifecycle and stage progress of document ingestion jobs.
    Maintains in-memory state with real-time MongoDB Atlas synchronization.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _persist_to_mongo(self, job: Dict[str, Any]):
        try:
            m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            m_client[MONGO_DB]["ingestion_jobs"].update_one(
                {"job_id": job["job_id"]},
                {"$set": job},
                upsert=True
            )
            m_client.close()
        except Exception as me:
            logger.warning(f"Could not persist job {job.get('job_id')} to MongoDB: {me}")

    def init_job(self, job_id: str, file_name: str) -> Dict[str, Any]:
        suffix = file_name[file_name.rfind("."):].lower() if "." in file_name else ""
        is_excel = suffix in [".xlsx", ".xls"]

        if is_excel:
            stages = [
                {"id": "upload", "name": "File Upload & Validation", "status": "completed", "detail": "File accepted"},
                {"id": "tabular", "name": "Multi-Sheet Extraction & Chunking", "status": "pending", "detail": "Waiting..."},
                {"id": "vectorize", "name": "Vector Embedding & Atlas Ingestion", "status": "pending", "detail": "Waiting..."}
            ]
        else:
            stages = [
                {"id": "upload", "name": "File Upload & Validation", "status": "completed", "detail": "File accepted"},
                {"id": "convert", "name": "Headless LibreOffice PDF Conversion", "status": "pending", "detail": "Waiting..."},
                {"id": "render", "name": "150 DPI Page Screenshot Extraction", "status": "pending", "detail": "Waiting..."},
                {"id": "vision", "name": "Gemma-3-27b-it Relevance Filter", "status": "pending", "detail": "Waiting..."},
                {"id": "extract", "name": "LlamaParse Structured Markdown Extraction", "status": "pending", "detail": "Waiting..."},
                {"id": "vectorize", "name": "Vector Embedding & Atlas Ingestion", "status": "pending", "detail": "Waiting..."}
            ]

        job = {
            "job_id": job_id,
            "file_name": file_name,
            "status": "in_progress",
            "progress_percent": 10,
            "current_stage": stages[1]["name"] if len(stages) > 1 else "Processing",
            "stages": stages,
            "chunks_indexed": 0,
            "error": None,
            "error_details": None,
            "started_at": time.time(),
            "completed_at": None
        }

        with self._lock:
            self._jobs[job_id] = job

        self._persist_to_mongo(job)
        return job

    def update_stage(self, job_id: str, stage_id: str, stage_status: str, detail: Optional[str] = None, progress: Optional[int] = None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            for stage in job["stages"]:
                if stage["id"] == stage_id:
                    stage["status"] = stage_status
                    if detail:
                        stage["detail"] = detail
                    if stage_status == "in_progress":
                        job["current_stage"] = stage["name"]
                elif stage_status == "in_progress" and stage["status"] == "pending":
                    pass

            if progress is not None:
                job["progress_percent"] = progress

            job_copy = dict(job)

        self._persist_to_mongo(job_copy)

    def fail_job(self, job_id: str, stage_id: str, error_msg: str, exc: Optional[Exception] = None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "failed"
            job["error"] = error_msg
            job["error_details"] = traceback.format_exc() if exc else error_msg
            job["completed_at"] = time.time()
            for stage in job["stages"]:
                if stage["id"] == stage_id:
                    stage["status"] = "failed"
                    stage["detail"] = f"Error: {error_msg}"
                    break
            job_copy = dict(job)

        self._persist_to_mongo(job_copy)

    def complete_job(self, job_id: str, chunks_indexed: int):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "completed"
            job["progress_percent"] = 100
            job["current_stage"] = "Completed Successfully"
            job["chunks_indexed"] = chunks_indexed
            job["completed_at"] = time.time()
            for stage in job["stages"]:
                if stage["status"] != "failed":
                    stage["status"] = "completed"
            job_copy = dict(job)

        self._persist_to_mongo(job_copy)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]

        # Check MongoDB
        try:
            m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            doc = m_client[MONGO_DB]["ingestion_jobs"].find_one({"job_id": job_id}, {"_id": 0})
            m_client.close()
            return doc
        except Exception:
            return None

    def list_jobs(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            job_list = list(self._jobs.values())
        
        # If in-memory is empty, pull recent from MongoDB
        if not job_list:
            try:
                m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
                cursor = m_client[MONGO_DB]["ingestion_jobs"].find({}, {"_id": 0}).sort("started_at", -1).limit(limit)
                job_list = list(cursor)
                m_client.close()
            except Exception:
                pass
        else:
            job_list.sort(key=lambda x: x.get("started_at", 0), reverse=True)
            job_list = job_list[:limit]

        return job_list

    def sweep_stale_jobs(self):
        """Sweeps any dangling 'in_progress' jobs from previous server runs and marks them as interrupted."""
        try:
            m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            result = m_client[MONGO_DB]["ingestion_jobs"].update_many(
                {"status": "in_progress"},
                {"$set": {
                    "status": "failed",
                    "error": "Pipeline process was interrupted by a service restart.",
                    "completed_at": time.time()
                }}
            )
            if result.modified_count > 0:
                logger.info(f"[JobManager] Swept {result.modified_count} stale in-progress jobs on startup.")
            m_client.close()
        except Exception as e:
            logger.warning(f"[JobManager] Could not sweep stale jobs: {e}")

# Global singleton instance
job_manager = JobManager()
