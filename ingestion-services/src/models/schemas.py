from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class JobStage(BaseModel):
    id: str
    name: str
    status: str = "pending"
    detail: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    file_name: str
    status: str = "in_progress"
    progress_percent: int = 0
    current_stage: str = "Initializing"
    stages: List[JobStage] = []
    chunks_indexed: int = 0
    error: Optional[str] = None
    error_details: Optional[str] = None
    started_at: float
    completed_at: Optional[float] = None

class UploadResponse(BaseModel):
    job_id: str
    file_name: str
    status: str
    message: str

class DocumentItem(BaseModel):
    file_name: str
    source_type: str
    chunks: int
    screenshot_url: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total: int

class HealthResponse(BaseModel):
    status: str
    service: str
    database_connected: bool
    vectors_ingested_count: int = 0
    r2_storage_configured: bool = False
    error: Optional[str] = None
