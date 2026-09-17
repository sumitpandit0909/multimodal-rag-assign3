import os
import logging
from pathlib import Path
from typing import Optional
from llama_parse import LlamaParse

logger = logging.getLogger(__name__)

def parse_page_with_llamaparse(file_path: Path, api_key: Optional[str] = None) -> Optional[str]:
    """
    Parses a document or image slice using LlamaParse into structured Markdown.
    Returns markdown string, or None if no API key is provided or error occurs.
    """
    api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        return None

    try:
        logger.info(f"Submitting {file_path.name} to LlamaParse (result_type='markdown')...")
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False
        )
        documents = parser.load_data(str(file_path))
        markdown_text = "\n\n".join([doc.text for doc in documents if doc.text])
        logger.info(f"LlamaParse extraction complete ({len(markdown_text)} chars).")
        return markdown_text
    except Exception as e:
        logger.warning(f"LlamaParse execution error on {file_path.name}: {e}")
        return None
