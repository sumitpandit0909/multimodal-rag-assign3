import logging
from typing import List, Dict, Any, Optional
from google import genai
from motor.motor_asyncio import AsyncIOMotorDatabase
from langgraph.graph import StateGraph, START, END

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from agent.state import AgentState
from agent.nodes import RAGNodes

logger = logging.getLogger(__name__)

def build_agent_graph(db: AsyncIOMotorDatabase, client: genai.Client):
    """
    Constructs and compiles the Self-Corrective Agentic RAG StateGraph:
      START -> retrieve -> grade_documents
                 ├── (relevant) -> generate -> END
                 ├── (irrelevant & rewrite_count < 1) -> rewrite_query -> retrieve
                 └── (irrelevant & rewrite_count >= 1 & no sources) -> no_sources -> END
    """
    nodes = RAGNodes(db, client)
    workflow = StateGraph(AgentState)

    # 1. Register Graph Nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("no_sources", nodes.no_sources)

    # 2. Register Transitions & Conditional Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "no_sources": "no_sources"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("no_sources", END)

    return workflow.compile()


# Graph singleton cache
_COMPILED_GRAPH = None

def get_compiled_graph(db: AsyncIOMotorDatabase, client: genai.Client):
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_agent_graph(db, client)
    return _COMPILED_GRAPH

@traceable(name="run_agentic_rag", run_type="chain")
async def run_agentic_rag(
    query: str, 
    history: List[Dict[str, Any]], 
    db: AsyncIOMotorDatabase, 
    client: genai.Client
) -> Dict[str, Any]:
    """
    Executes the compiled LangGraph workflow asynchronously.
    """
    graph = get_compiled_graph(db, client)
    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "history": history,
        "retrieved_sources": [],
        "documents_relevant": False,
        "rewrite_count": 0,
        "answer": "",
        "source_nodes": []
    }

    final_state = await graph.ainvoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "source_nodes": final_state.get("source_nodes", [])
    }
