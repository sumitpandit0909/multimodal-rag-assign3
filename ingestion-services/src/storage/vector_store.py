from google import genai
from pymongo import MongoClient
from typing import List, Dict

from google.genai import types

def generate_embeddings(client: genai.Client, texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings

def ingest_to_atlas(mongo_uri: str, db_name: str, records: List[Dict], genai_client: genai.Client):
    mongo_client = MongoClient(mongo_uri)
    coll = mongo_client[db_name]["vectors"]

    texts = [r["text_content"] for r in records]
    vectors = generate_embeddings(genai_client, texts)

    docs = []
    for record, vector in zip(records, vectors):
        doc = {**record, "embedding": vector}
        docs.append(doc)

    if docs:
        coll.insert_many(docs)
    mongo_client.close()
