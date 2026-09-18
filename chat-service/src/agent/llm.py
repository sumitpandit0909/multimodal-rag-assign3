import os
import logging
from typing import Optional
from openai import OpenAI
from google import genai
from google.genai import types

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
    """Returns a LangSmith-wrapped OpenAI client configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    raw_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return wrap_openai(raw_client)

@traceable(name="gemma_chat_completion", run_type="llm")
def generate_with_gemma_openrouter(
    client: OpenAI, 
    prompt: str, 
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """Generates completion using google/gemma-3-27b-it via OpenRouter."""
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content or ""

def generate_with_gemini_fallback(
    client: genai.Client, 
    contents, 
    config: Optional[types.GenerateContentConfig] = None
) -> str:
    """Fallback generator trying Gemini models with exponential retry."""
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None
    
    cfg = config or types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2
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
