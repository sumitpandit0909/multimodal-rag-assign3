import os
import logging
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Enterprise Multimodal Knowledge Assistant.
Answer the user's inquiry based strictly on the provided context retrieved from corporate files.
Be concise, direct, and factual. Limit your response to 2-3 focused paragraphs.
When you cite information, mark the source using inline brackets like [1], [2] matching the provided sources index.
If you do not find the answer in the provided context, state that clearly without guessing and DO NOT cite any sources.
"""

def get_openrouter_api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY")

def get_generator_llm() -> Optional[ChatOpenAI]:
    """
    Native LangChain ChatOpenAI configured for OpenRouter google/gemma-3-27b-it.
    Configured with a 14-second fail-fast timeout and 500 max_tokens to eliminate latency spikes.
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        return None
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.2,
        max_tokens=500,
        request_timeout=14.0,
        max_retries=1
    )

def get_utility_llm() -> Optional[ChatOpenAI]:
    """
    Native LangChain ChatOpenAI configured for OpenRouter google/gemini-2.5-flash-lite.
    Ultra-low latency for relevance grading and single-line query rewrites (<250ms).
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        return None
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.0,
        max_tokens=35,
        request_timeout=6.0,
        max_retries=1
    )

async def ainvoke_generator(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Asynchronously calls the LangChain generator model with OpenRouter."""
    llm = get_generator_llm()
    if not llm:
        return ""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    resp = await llm.ainvoke(messages)
    return str(resp.content or "")

async def ainvoke_utility(prompt: str, system_prompt: str = "Output ONLY the requested string.") -> str:
    """Asynchronously calls the LangChain fast utility model."""
    llm = get_utility_llm()
    if not llm:
        return ""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    resp = await llm.ainvoke(messages)
    return str(resp.content or "")

def generate_with_gemini_fallback(
    client: genai.Client, 
    contents, 
    config: Optional[types.GenerateContentConfig] = None
) -> str:
    """Direct Google GenAI SDK fallback with exponential retry if OpenRouter times out."""
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None
    
    cfg = config or types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=500
    )

    for model_name in models_to_try:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=cfg
            )
            return resp.text or ""
        except Exception as e:
            err_str = str(e).upper()
            if any(k in err_str for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "QUOTA"]):
                last_err = e
                continue
            raise e

    if last_err:
        raise last_err
    return ""
