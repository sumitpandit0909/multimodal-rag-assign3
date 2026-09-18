import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

from core.config import (
    MONGO_URI, MONGO_DB, OUTPUT_DIR, LLAMA_CLOUD_API_KEY, genai_client
)
from services.job_manager import job_manager
from pipelines.converter import convert_to_pdf, parse_pptx_fallback
from pipelines.visual_pipeline import extract_page_screenshots
from pipelines.vision_filter import classify_page_image, extract_image_description
from pipelines.llama_parser import parse_page_with_llamaparse
from pipelines.excel_pipeline import parse_excel_file
from pipelines.chunking import recursive_chunk_text
from storage.vector_store import ingest_to_atlas
from storage.r2_storage import r2_storage

logger = logging.getLogger(__name__)

def _process_single_page(
    sc: Dict[str, Any],
    file_path: Path,
    client: genai.Client
) -> List[Dict[str, Any]]:
    """
    Worker function executed in parallel for a single page screenshot.
    Performs:
    1. Gemma-3-27b-it vision classification filter
    2. LlamaParse layout extraction (or fallback)
    3. Cloudflare R2 image upload (or local fallback)
    4. Recursive character text chunking (800 chars, 150 overlap)
    """
    page_num = sc["page_number"]
    img_path = sc["image_path"]

    # 1. Vision Relevance Filter
    is_valid = classify_page_image(client, img_path)
    if not is_valid:
        logger.info(f"Filtered out {file_path.name} page {page_num} (no actionable knowledge).")
        return []

    # 2. Content Extraction (LlamaParse or OCR fallback)
    markdown_content = parse_page_with_llamaparse(img_path, LLAMA_CLOUD_API_KEY)
    if not markdown_content:
        extracted_text = sc.get("text", "")
        if len(extracted_text) < 25:
            extracted_text = extract_image_description(client, img_path)
        markdown_content = extracted_text

    # 3. Cloudflare R2 Upload (with fallback to local static URL)
    screenshot_url = f"/static/images/{file_path.stem}/{img_path.name}"
    if r2_storage.is_configured():
        r2_key = f"images/{file_path.stem}/{img_path.name}"
        uploaded_url = r2_storage.upload_file(img_path, r2_key, "image/png")
        if uploaded_url:
            screenshot_url = uploaded_url

    # 4. Recursive Character Chunking (800 chars, 150 overlap)
    sub_chunks = recursive_chunk_text(markdown_content, chunk_size=800, chunk_overlap=150)
    if not sub_chunks:
        sub_chunks = [markdown_content]

    page_records = []
    for chunk_text in sub_chunks:
        content = f"### Document: {file_path.name} | Page {page_num}\n{chunk_text}"
        page_records.append({
            "file_name": file_path.name,
            "source_type": "visual",
            "page_number": page_num,
            "sheet_name": None,
            "screenshot_url": screenshot_url,
            "raw_data": None,
            "text_content": content
        })
    return page_records

def process_file(file_path: Path, client: genai.Client = genai_client, job_id: Optional[str] = None):
    """
    Modular pipeline orchestrator for visual documents and spreadsheets.
    Applies concurrency for page classification, extraction, and cloud storage.
    """
    if not job_id:
        job_id = str(uuid.uuid4())
        job_manager.init_job(job_id, file_path.name)

    logger.info(f"[Job {job_id}] Processing file: {file_path.name}")
    suffix = file_path.suffix.lower()
    records: List[Dict[str, Any]] = []

    try:
        if suffix in [".xlsx", ".xls"]:
            # Path B: Tabular Documents
            logger.info(f"[Job {job_id}] Running Path B (Tabular Multi-Sheet Extraction) for: {file_path.name}")
            job_manager.update_stage(job_id, "tabular", "in_progress", "Parsing Excel sheets and batching rows...", progress=25)
            
            records = parse_excel_file(file_path)
            if not records:
                job_manager.fail_job(job_id, "tabular", f"No readable rows or sheets found in {file_path.name}")
                return

            job_manager.update_stage(job_id, "tabular", "completed", f"Extracted {len(records)} row chunks", progress=60)

        elif suffix in [".pdf", ".docx", ".doc", ".ppt", ".pptx"]:
            # Path A: Visual Documents
            logger.info(f"[Job {job_id}] Running Path A (Visual Conversion & Concurrent Processing) for: {file_path.name}")
            pdf_dir = OUTPUT_DIR / "pdfs"
            images_dir = OUTPUT_DIR / "images" / file_path.stem
            pdf_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            # 1. Convert to PDF if DOCX/PPT
            job_manager.update_stage(job_id, "convert", "in_progress", "Converting document to PDF...", progress=20)
            try:
                pdf_path = convert_to_pdf(file_path, pdf_dir)
                job_manager.update_stage(job_id, "convert", "completed", f"PDF generated: {pdf_path.name}", progress=35)
                if r2_storage.is_configured():
                    try:
                        r2_storage.upload_file(pdf_path, f"pdfs/{pdf_path.name}", "application/pdf")
                    except Exception as r2_pdf_err:
                        logger.warning(f"Failed to upload PDF to R2: {r2_pdf_err}")
            except Exception as conv_err:
                if suffix in [".pptx", ".ppt"]:
                    logger.warning(f"LibreOffice failed ({conv_err}), falling back to python-pptx extraction...")
                    fallback_slides = parse_pptx_fallback(file_path)
                    if fallback_slides:
                        job_manager.update_stage(job_id, "convert", "completed", f"Extracted {len(fallback_slides)} slides via PPTX parser", progress=40)
                        job_manager.update_stage(job_id, "render", "completed", "Extracted slide structure (fallback mode)", progress=55)
                        job_manager.update_stage(job_id, "vision", "completed", f"Processed {len(fallback_slides)} presentation slides", progress=70)
                        job_manager.update_stage(job_id, "extract", "completed", f"Extracted text from {len(fallback_slides)} slides", progress=85)

                        for s in fallback_slides:
                            slide_chunks = recursive_chunk_text(s["text"], chunk_size=800, chunk_overlap=150)
                            if not slide_chunks:
                                slide_chunks = [s["text"]]
                            for chunk_text in slide_chunks:
                                records.append({
                                    "file_name": file_path.name,
                                    "source_type": "visual",
                                    "page_number": s["page_number"],
                                    "sheet_name": None,
                                    "screenshot_url": None,
                                    "raw_data": None,
                                    "text_content": f"### Document: {file_path.name} | Slide {s['page_number']}\n{chunk_text}"
                                })

                        # Proceed directly to vectorization
                        if records:
                            job_manager.update_stage(job_id, "vectorize", "in_progress", f"Generating 768d embeddings for {len(records)} chunks...", progress=90)
                            try:
                                ingest_to_atlas(MONGO_URI, MONGO_DB, records, client)
                                job_manager.update_stage(job_id, "vectorize", "completed", f"Indexed {len(records)} chunks into Atlas", progress=100)
                                job_manager.complete_job(job_id, len(records))
                                logger.info(f"[Job {job_id}] Ingestion completed successfully.")
                                return
                            except Exception as vect_err:
                                job_manager.fail_job(job_id, "vectorize", f"Embedding or MongoDB indexing failed: {str(vect_err)}", vect_err)
                                return
                job_manager.fail_job(job_id, "convert", f"LibreOffice conversion failed: {str(conv_err)}", conv_err)
                return

            # 2. Extract page screenshots
            job_manager.update_stage(job_id, "render", "in_progress", "Rendering 150 DPI page screenshots...", progress=40)
            try:
                screenshots = extract_page_screenshots(pdf_path, images_dir)
                total_pages = len(screenshots)
                job_manager.update_stage(job_id, "render", "completed", f"Rendered {total_pages} page screenshots", progress=50)
            except Exception as rend_err:
                job_manager.fail_job(job_id, "render", f"PyMuPDF screenshotting failed: {str(rend_err)}", rend_err)
                return

            # 3 & 4. Concurrent Page Classification, Extraction & R2 Upload
            job_manager.update_stage(job_id, "vision", "in_progress", f"Analyzing {total_pages} pages in parallel (4 workers)...", progress=55)
            job_manager.update_stage(job_id, "extract", "in_progress", "Extracting structured markdown...", progress=60)

            completed_count = 0
            # Run up to 4 pages concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_process_single_page, sc, file_path, client): sc for sc in screenshots}
                for future in as_completed(futures):
                    completed_count += 1
                    current_progress = 55 + int((completed_count / total_pages) * 30)
                    job_manager.update_stage(
                        job_id, "extract", "in_progress",
                        f"Processed page {completed_count}/{total_pages}...",
                        progress=min(current_progress, 88)
                    )
                    try:
                        page_records = future.result()
                        if page_records:
                            records.extend(page_records)
                    except Exception as page_err:
                        logger.warning(f"Error processing page in thread pool: {page_err}")

            # Sort records by page number
            records.sort(key=lambda r: r.get("page_number") or 0)
            job_manager.update_stage(job_id, "vision", "completed", f"Evaluated all {total_pages} pages", progress=75)
            job_manager.update_stage(job_id, "extract", "completed", f"Extracted {len(records)} useful page records", progress=85)

        else:
            job_manager.fail_job(job_id, "upload", f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .ppt, .pptx, .xlsx, .xls")
            return

        # Vector Ingestion
        if records:
            job_manager.update_stage(job_id, "vectorize", "in_progress", f"Generating 768d embeddings for {len(records)} chunks...", progress=90)
            try:
                ingest_to_atlas(MONGO_URI, MONGO_DB, records, client)
                job_manager.update_stage(job_id, "vectorize", "completed", f"Indexed {len(records)} chunks into Atlas", progress=100)
                job_manager.complete_job(job_id, len(records))
                logger.info(f"[Job {job_id}] Ingestion completed successfully.")
            except Exception as vect_err:
                job_manager.fail_job(job_id, "vectorize", f"Embedding or MongoDB indexing failed: {str(vect_err)}", vect_err)
        else:
            job_manager.fail_job(job_id, "extract", "No valid content or pages were extracted from document.")

    except Exception as e:
        logger.error(f"[Job {job_id}] Pipeline exception: {e}", exc_info=True)
        job_manager.fail_job(job_id, "upload", f"Unexpected ingestion failure: {str(e)}", e)
