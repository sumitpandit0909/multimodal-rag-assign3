from google import genai
from google.genai import types
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = (
    "Analyze this page image from a presentation or document. "
    "Does it contain useful knowledge, text, diagram, table, or facts? "
    "Answer strictly with either 'YES' or 'NO'."
)

def classify_page_image(client: genai.Client, image_path: Path) -> bool:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            CLASSIFICATION_PROMPT
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=100,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    result = (response.text or "").strip().upper()
    logger.info(f"Image {image_path.name} classification: {result}")
    return "YES" in result

def extract_image_description(client: genai.Client, image_path: Path) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Transcribe and summarize all key text, facts, tables, and topics shown on this page/slide for search indexing."
        ],
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text.strip()

