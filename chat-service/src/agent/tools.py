from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

async def search_vector_store(query: str, db: AsyncIOMotorDatabase, genai_client: genai.Client, top_k: int = 4) -> List[Dict]:
    try:
        # 1. Embed user query with gemini-embedding-001 and 768 dimensions
        embed_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        query_vector = embed_response.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []

    # 2. Vector Search Aggregation Pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
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
        return await cursor.to_list(length=top_k)
    except Exception as e:
        logger.warning(f"Vector search failed (index 'vector_index' may not yet be active in Atlas): {e}")
        return []

