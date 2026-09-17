import os
import json
import logging
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from openai import OpenAI

try:
    from langsmith.wrappers import wrap_openai
    from langsmith import traceable
except ImportError:
    wrap_openai = lambda c: c
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Enterprise Multimodal Knowledge Assistant.
Answer the user's inquiry based strictly on the provided context retrieved from corporate files.
When you cite information, mark the source using inline brackets like [1], [2] matching the provided sources index.
If you do not find the answer in the provided context, state that clearly without guessing.
"""

def get_openrouter_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    raw_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return wrap_openai(raw_client)

@traceable(name="gemma_chat_completion", run_type="llm")
def generate_with_gemma_openrouter(client: OpenAI, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2048
    )
    return response.choices[0].message.content or ""

def generate_with_fallback(client: genai.Client, contents, config):
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None
    for model_name in models_to_try:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e).upper()
            if any(k in err_str for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "QUOTA"]):
                last_err = e
                continue
            raise e
    if last_err:
        raise last_err

@traceable(name="agentic_rag_execution", run_type="chain")
async def run_agentic_rag(query: str, history: List[Dict], retrieved_sources: List[Dict], client: genai.Client) -> Dict:
    openrouter_client = get_openrouter_client()

    if not retrieved_sources:
        no_sources_prompt = (
            f"User asked: '{query}'\n"
            "Inform the user politely that no matching documents or vector records were found in the enterprise knowledge base. "
            "If documents have not been ingested yet, invite them to drop files into the ingestion service."
        )
        if openrouter_client:
            try:
                answer = generate_with_gemma_openrouter(openrouter_client, no_sources_prompt)
                return {"answer": answer, "source_nodes": []}
            except Exception as e:
                logger.warning(f"OpenRouter Gemma error on no sources: {e}, falling back to Gemini")

        response = generate_with_fallback(
            client=client,
            contents=no_sources_prompt,
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

    answer_text = ""
    if openrouter_client:
        try:
            logger.info("Generating answer using google/gemma-3-27b-it via OpenRouter...")
            answer_text = generate_with_gemma_openrouter(openrouter_client, user_prompt)
        except Exception as e:
            logger.warning(f"OpenRouter Gemma generation failed: {e}, falling back to Gemini...")

    if not answer_text:
        response = generate_with_fallback(
            client=client,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2
            )
        )
        answer_text = response.text

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
        "answer": answer_text,
        "source_nodes": source_nodes
    }
