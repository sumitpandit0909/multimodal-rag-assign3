import os
import base64
import logging
from pathlib import Path
from typing import Optional
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

CLASSIFICATION_PROMPT = (
    "Analyze this page image from a presentation or document. "
    "Does it contain useful knowledge, text, diagram, table, or facts? "
    "Answer strictly with either 'YES' or 'NO'."
)

def get_openrouter_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    raw_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return wrap_openai(raw_client)

@traceable(name="gemma_vision_filter", run_type="llm")
def classify_page_image_openrouter(client: OpenAI, image_path: Path) -> bool:
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_data}"
                        }
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=20
    )
    
    content = response.choices[0].message.content or ""
    result = content.strip().upper()
    logger.info(f"[Gemma-3-27b-it Vision] Image {image_path.name} classification: {result}")
    return "YES" in result

def classify_page_image(genai_client, image_path: Path) -> bool:
    openrouter_client = get_openrouter_client()
    if openrouter_client:
        try:
            return classify_page_image_openrouter(openrouter_client, image_path)
        except Exception as e:
            logger.warning(f"OpenRouter Gemma-3 vision failed ({e}), falling back to Gemini/default: {e}")

    # Fallback to Gemini if available
    if genai_client:
        try:
            from google.genai import types
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            for model_name in ["gemini-2.0-flash", "gemini-2.5-flash"]:
                try:
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                            CLASSIFICATION_PROMPT
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=20,
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                    result = (response.text or "").strip().upper()
                    logger.info(f"[{model_name} Vision] Image {image_path.name} classification: {result}")
                    return "YES" in result
                except Exception as ex:
                    logger.warning(f"Gemini model {model_name} vision error: {ex}")
                    continue
        except Exception as fallback_err:
            logger.error(f"Fallback vision classification failed: {fallback_err}")
            
    # Default to accepting page so documents are not dropped
    return True

@traceable(name="gemma_extract_description", run_type="llm")
def extract_image_description(genai_client, image_path: Path) -> str:
    openrouter_client = get_openrouter_client()
    if openrouter_client:
        try:
            with open(image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            response = openrouter_client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe and summarize all key text, facts, tables, and topics shown on this page/slide for search indexing."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenRouter description extraction error: {e}")

    # Fallback to Gemini
    if genai_client:
        try:
            from google.genai import types
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    "Transcribe and summarize all key text, facts, tables, and topics shown on this page/slide for search indexing."
                ],
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini description extraction error: {e}")

    return ""
