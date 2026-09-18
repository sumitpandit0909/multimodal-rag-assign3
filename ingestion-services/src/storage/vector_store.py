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
def generate_embeddings(client: genai.Client, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Generates 768d vector embeddings using gemini-embedding-001 with batching and exponential retry.
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_success = False
        last_err = None

        # Attempt batch embedding
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                for emb in response.embeddings:
                    embeddings.append(emb.values)
                batch_success = True
                break
            except Exception as e:
                last_err = e
                logger.warning(f"Batch embedding attempt {attempt + 1} failed: {e}. Retrying in 1s...")
                time.sleep(1.0)

        # Fallback to single item embedding if batch failed
        if not batch_success:
            logger.warning(f"Batch embedding failed ({last_err}), falling back to single-item embedding for batch...")
            for text in batch:
                item_success = False
                for attempt in range(3):
                    try:
                        response = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=text,
                            config=types.EmbedContentConfig(output_dimensionality=768)
                        )
                        embeddings.append(response.embeddings[0].values)
                        item_success = True
                        break
                    except Exception as e:
                        time.sleep(1.0)
                if not item_success:
                    raise RuntimeError(f"Failed to generate embedding after 3 retries: {last_err}")

    return embeddings

@traceable(name="ingest_to_atlas", run_type="tool")
def ingest_to_atlas(mongo_uri: str, db_name: str, records: List[Dict], genai_client: genai.Client):
    """
    Purges stale vectors for the file to prevent duplicates, generates 768d embeddings in batches,
    and inserts clean documents into MongoDB Atlas collection 'vectors'.
    """
    if not records:
        return

    mongo_client = MongoClient(mongo_uri)
    coll = mongo_client[db_name]["vectors"]

    # 1. Automatic File De-duplication: purge previous chunks for this file
    file_name = records[0].get("file_name")
    if file_name:
        del_res = coll.delete_many({"file_name": file_name})
        if del_res.deleted_count > 0:
            logger.info(f"Purged {del_res.deleted_count} previous chunks for '{file_name}' to prevent vector duplicates.")

    # 2. Batch Embedding Generation
    texts = [r["text_content"] for r in records]
    vectors = generate_embeddings(genai_client, texts)

    # 3. Clean and insert
    docs = []
    for record, vector in zip(records, vectors):
        doc = clean_doc_for_mongo({**record, "embedding": vector})
        docs.append(doc)

    if docs:
        coll.insert_many(docs)
        logger.info(f"Successfully inserted {len(docs)} chunks into MongoDB Atlas vectors collection.")

    mongo_client.close()
