import os
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict
import logging

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

logger = logging.getLogger(__name__)

# In-memory LRU cache for query embeddings to avoid duplicate remote embedding calls
_QUERY_EMBEDDING_CACHE: Dict[str, List[float]] = {}

@traceable(name="search_vector_store", run_type="retriever")
async def search_vector_store(query: str, db: AsyncIOMotorDatabase, genai_client: genai.Client, top_k: int = 8) -> List[Dict]:
    cache_key = query.strip().lower()
    
    if cache_key in _QUERY_EMBEDDING_CACHE:
        query_vector = _QUERY_EMBEDDING_CACHE[cache_key]
        logger.info(f"[Retriever] Cache HIT for query embedding: '{cache_key[:30]}...'")
    else:
        try:
            embed_response = genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=query,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            query_vector = embed_response.embeddings[0].values
            if len(_QUERY_EMBEDDING_CACHE) > 1000:
                _QUERY_EMBEDDING_CACHE.clear()
            _QUERY_EMBEDDING_CACHE[cache_key] = query_vector
            logger.info(f"[Retriever] Generated new query embedding for '{cache_key[:30]}...'")
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    # Vector Search Aggregation Pipeline in MongoDB Atlas
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, top_k * 10),
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "file_name": 1,
                "source_type": 1,
                "page_number": 1,
                "sheet_name": 1,
                "screenshot_url": 1,
                "raw_data": 1,
                "text_content": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    try:
        cursor = db["vectors"].aggregate(pipeline)
        results = await cursor.to_list(length=top_k)
        if results:
            top_score = results[0].get("score", 0.0)
            logger.info(f"[Retriever] Retrieved {len(results)} chunks. Top vector score: {top_score:.4f}")
        return results
    except Exception as e:
        logger.warning(f"Vector search failed (index 'vector_index' may not yet be active in Atlas): {e}")
        return []
