from typing import List, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Typed state dictionary passed between nodes in the LangGraph Agentic RAG graph.
    """
    query: str                  # Current search query (can be modified by rewrite_query)
    original_query: str         # Original query as entered by the user
    history: List[Dict[str, Any]]# Past conversational turns for the session
    retrieved_sources: List[Dict[str, Any]] # Raw chunks retrieved from MongoDB Atlas
    documents_relevant: bool    # Grader evaluation flag
    rewrite_count: int          # Loop iteration counter (bounded to prevent infinite loops)
    answer: str                 # Final synthesized answer with inline citation tags
    source_nodes: List[Dict[str, Any]] # Structured citation metadata for UI rendering
