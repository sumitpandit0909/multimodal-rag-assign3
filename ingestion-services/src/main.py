import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env from root assignment3 folder and current folder
load_dotenv()
load_dotenv(SRC_DIR.parent.parent / ".env")

from pipelines.converter import convert_to_pdf
from pipelines.visual_pipeline import extract_page_screenshots
from pipelines.vision_filter import classify_page_image, extract_image_description
from pipelines.excel_pipeline import parse_excel_file
from storage.vector_store import ingest_to_atlas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion-service")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "multimodal_rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DATA_DIR = Path(os.getenv("DATA_DIR", SRC_DIR.parent.parent / "data_drop"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", SRC_DIR.parent.parent / "processed_output"))

def process_file(file_path: Path, genai_client: genai.Client):
    logger.info(f"Processing file: {file_path.name}")
    suffix = file_path.suffix.lower()
    records = []

    if suffix in [".xlsx", ".xls"]:
        # Path B: Excel Files
        logger.info(f"Running Path B (Excel Tabular) for: {file_path.name}")
        records = parse_excel_file(file_path)

    elif suffix in [".pdf", ".docx", ".doc", ".ppt", ".pptx"]:
        # Path A: Visual Documents
        logger.info(f"Running Path A (Visual Conversion & Screenshotting) for: {file_path.name}")
        pdf_dir = OUTPUT_DIR / "pdfs"
        images_dir = OUTPUT_DIR / "images" / file_path.stem
        pdf_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # 1. Convert to PDF if DOCX/PPT
        pdf_path = convert_to_pdf(file_path, pdf_dir)

        # 2. Extract page screenshots
        screenshots = extract_page_screenshots(pdf_path, images_dir)

        # 3. Classify with Gemini Vision ("YES/NO")
        for sc in screenshots:
            is_valid = classify_page_image(genai_client, sc["image_path"])
            if not is_valid:
                logger.info(f"Filtered out page {sc['page_number']} (no useful knowledge).")
                continue

            extracted_text = sc.get("text", "")
            if len(extracted_text) < 25:
                extracted_text = extract_image_description(genai_client, sc["image_path"])

            screenshot_url = f"/static/images/{file_path.stem}/{sc['image_path'].name}"
            content = f"### Document: {file_path.name} | Page {sc['page_number']}\n{extracted_text}"

            records.append({
                "file_name": file_path.name,
                "source_type": "visual",
                "page_number": sc["page_number"],
                "sheet_name": None,
                "screenshot_url": screenshot_url,
                "raw_data": None,
                "text_content": content
            })
    else:
        logger.warning(f"Unsupported file extension: {suffix}")
        return

    if records:
        logger.info(f"Ingesting {len(records)} chunks into MongoDB Atlas...")
        ingest_to_atlas(MONGO_URI, MONGO_DB, records, genai_client)
        logger.info("Ingestion completed successfully.")

import time

def main():
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Gemini API calls may fail.")
    genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingestion worker started. Listening for files in {DATA_DIR}...")
    processed_files = set()

    # If run in one-shot batch mode
    is_daemon = os.getenv("DAEMON_MODE", "true").lower() == "true"

    while True:
        files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
        for file in files:
            file_key = f"{file.name}_{file.stat().st_mtime}"
            if file_key not in processed_files:
                try:
                    process_file(file, genai_client)
                    processed_files.add(file_key)
                except Exception as e:
                    logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
        
        if not is_daemon:
            break
        time.sleep(3)

if __name__ == "__main__":
    main()

