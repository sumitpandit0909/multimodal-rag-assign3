from typing import List, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Typed state dictionary passed between nodes in the LangGraph Agentic RAG graph.
    """
    query: str                             # User inquiry
    history: List[Dict[str, Any]]          # Past conversational turns for the session
    retrieved_sources: List[Dict[str, Any]]# Raw chunks retrieved from MongoDB Atlas
    documents_relevant: bool               # Grader evaluation flag
    answer: str                            # Final synthesized answer with inline citation tags
    source_nodes: List[Dict[str, Any]]     # Structured citation metadata for UI rendering
