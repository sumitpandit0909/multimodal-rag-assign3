from typing import Optional, List, Dict, Any
from pydantic import BaseModel

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

class DocumentItem(BaseModel):
    file_name: str
    source_type: str
    chunks: int
    screenshot_url: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total: int

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    vectors_ingested_count: int
    ready_for_queries: bool
