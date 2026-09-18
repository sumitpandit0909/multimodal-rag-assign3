from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class RerankOutput(BaseModel):
    """Structured output for the Semantic Re-ranker & Relevance Grader."""
    is_relevant: bool = Field(
        description="True if any candidate passage contains facts or context to answer the user question; False if completely out-of-domain."
    )
    ranked_ids: List[int] = Field(
        default_factory=list,
        description="Top 1 to 3 candidate passage IDs in descending order of relevance (e.g., [2, 1, 5]). Empty list if not relevant."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief 1-sentence explanation of why these passages were selected or rejected."
    )

class SynthesisOutput(BaseModel):
    """Structured output for grounded answer generation with verified citations."""
    answer: str = Field(
        description="Grounded, factual answer citing sources inline using bracketed format like [1], [2]."
    )
    cited_sources: List[int] = Field(
        default_factory=list,
        description="Exact list of citation ID numbers actually referenced in the answer (e.g., [1, 2])."
    )

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
