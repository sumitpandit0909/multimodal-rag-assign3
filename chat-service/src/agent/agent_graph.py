from google import genai
from google.genai import types
from typing import List, Dict
import json

SYSTEM_PROMPT = """You are an Enterprise Multimodal Knowledge Assistant.
Answer the user's inquiry based strictly on the provided context retrieved from corporate files.
When you cite information, mark the source using inline brackets like [1], [2] matching the provided sources index.
If you do not find the answer in the provided context, state that clearly without guessing.
"""

async def run_agentic_rag(query: str, history: List[Dict], retrieved_sources: List[Dict], client: genai.Client) -> Dict:
    if not retrieved_sources:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"User asked: '{query}'\nInform the user politely that no matching documents or vector records were found in the enterprise knowledge base. If documents have not been ingested yet, invite them to drop files into the ingestion service.",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3
            )
        )
        return {
            "answer": response.text,
            "source_nodes": []
        }

    # Prepare formatted context
    context_str = ""
    for idx, s in enumerate(retrieved_sources, start=1):
        src_label = f"Page {s['page_number']}" if s.get("page_number") else f"Sheet {s.get('sheet_name')}"
        context_str += f"\n--- Source [{idx}]: {s['file_name']} ({src_label}) ---\n{s['text_content']}\n"

    user_prompt = f"User Question: {query}\n\nContext:\n{context_str}\n\nAnswer with inline citations [1], [2]:"

    # Call Gemini model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )

    # Format structured source nodes for frontend consumption
    source_nodes = []
    for idx, s in enumerate(retrieved_sources, start=1):
        node = {
            "citation_id": idx,
            "file_name": s["file_name"],
            "source_type": s["source_type"]
        }
        if s["source_type"] == "visual":
            node["screenshot_url"] = s.get("screenshot_url")
            node["page_number"] = s.get("page_number")
        elif s["source_type"] == "tabular":
            node["sheet_name"] = s.get("sheet_name")
            node["raw_data"] = s.get("raw_data")
        source_nodes.append(node)

    return {
        "answer": response.text,
        "source_nodes": source_nodes
    }
