import os
import time
import math
import logging
from google import genai
from google.genai import types
from pymongo import MongoClient
from typing import List, Dict, Any

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

logger = logging.getLogger(__name__)

def clean_doc_for_mongo(obj: Any) -> Any:
    """Recursively cleanse documents against NaN, inf, or non-serializable types."""
    if isinstance(obj, dict):
        return {str(k): clean_doc_for_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_doc_for_mongo(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj

@traceable(name="generate_embeddings", run_type="embedding")
def generate_embeddings(client: genai.Client, texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        success = False
        last_err = None
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                embeddings.append(response.embeddings[0].values)
                success = True
                break
            except Exception as e:
                last_err = e
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(1.5)
        if not success:
            raise RuntimeError(f"Failed to generate embedding for text after 3 retries: {last_err}")
    return embeddings

@traceable(name="ingest_to_atlas", run_type="tool")
def ingest_to_atlas(mongo_uri: str, db_name: str, records: List[Dict], genai_client: genai.Client):
    mongo_client = MongoClient(mongo_uri)
    coll = mongo_client[db_name]["vectors"]

    texts = [r["text_content"] for r in records]
    vectors = generate_embeddings(genai_client, texts)

    docs = []
    for record, vector in zip(records, vectors):
        doc = clean_doc_for_mongo({**record, "embedding": vector})
        docs.append(doc)

    if docs:
        coll.insert_many(docs)
    mongo_client.close()
