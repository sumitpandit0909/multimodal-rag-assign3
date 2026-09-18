import logging
from fastapi import APIRouter, status, HTTPException
from db.connection import get_db, get_genai_client, get_mongo_client
from db.memory import MongoChatMemory
from models.schemas import ChatRequest, ChatResponse, HealthResponse, DocumentListResponse
from agent.graph import run_agentic_rag

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat & Knowledge"])

@router.get("/health", status_code=status.HTTP_200_OK, response_model=HealthResponse)
async def health_check():
    """Health check endpoint validating MongoDB Atlas connectivity and vector counts."""
    db = get_db()
    mongo_client = get_mongo_client()
    try:
        await mongo_client.admin.command("ping")
        vector_count = await db["vectors"].count_documents({})
        return HealthResponse(
            status="healthy",
            database_connected=True,
            vectors_ingested_count=vector_count,
            ready_for_queries=vector_count > 0
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            database_connected=False,
            vectors_ingested_count=0,
            ready_for_queries=False
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Agentic RAG endpoint executing the compiled LangGraph state workflow.
    Handles multi-turn memory, vector search, grading, and answer synthesis.
    """
    db = get_db()
    genai_client = get_genai_client()
    memory = MongoChatMemory(db)
    
    # 1. Retrieve conversation history
    history = await memory.get_history(request.session_id)
    
    # 2. Run LangGraph Self-Corrective Agentic RAG workflow
    rag_result = await run_agentic_rag(
        query=request.message,
        history=history,
        db=db,
        client=genai_client
    )
    
    # 3. Persist conversation turn
    await memory.add_turn(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=rag_result["answer"],
        sources=rag_result["source_nodes"]
    )
    
    return ChatResponse(
        session_id=request.session_id,
        answer=rag_result["answer"],
        sources=rag_result["source_nodes"]
    )

@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """Returns an aggregated list of indexed documents and chunk statistics."""
    db = get_db()
    try:
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
        cursor = db["vectors"].aggregate(pipeline)
        docs = await cursor.to_list(length=100)
        return DocumentListResponse(documents=docs, total=len(docs))
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return DocumentListResponse(documents=[], total=0)
