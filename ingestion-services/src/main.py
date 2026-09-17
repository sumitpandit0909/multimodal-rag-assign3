import os
import sys
import time
import uuid
import shutil
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env from root and current directory
load_dotenv()
load_dotenv(SRC_DIR.parent.parent / ".env")

from fastapi import FastAPI, status, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pipelines.converter import convert_to_pdf
from pipelines.visual_pipeline import extract_page_screenshots
from pipelines.vision_filter import classify_page_image, extract_image_description
from pipelines.llama_parser import parse_page_with_llamaparse
from pipelines.excel_pipeline import parse_excel_file
from storage.vector_store import ingest_to_atlas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion-service")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "multimodal_rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

DATA_DIR = Path(os.getenv("DATA_DIR", SRC_DIR.parent.parent / "data_drop"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", SRC_DIR.parent.parent / "processed_output"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

app = FastAPI(title="Enterprise Multimodal RAG Ingestion Service")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve generated page screenshots
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")

# In-memory and persistent job tracker
JOBS: Dict[str, Dict[str, Any]] = {}

def get_job_stages_template(suffix: str) -> List[Dict[str, Any]]:
    if suffix in [".xlsx", ".xls"]:
        return [
            {"id": "upload", "name": "File Upload & Validation", "status": "completed", "detail": "File accepted"},
            {"id": "tabular", "name": "Excel Sheet & Rows Parsing", "status": "pending", "detail": "Extracting structured tabular data (No screenshots)"},
            {"id": "vectorize", "name": "Vector Embedding & MongoDB Atlas Indexing", "status": "pending", "detail": "Embedding rows and indexing into Atlas"}
        ]
    return [
        {"id": "upload", "name": "File Upload & Validation", "status": "completed", "detail": "File accepted"},
        {"id": "convert", "name": "LibreOffice Document Conversion", "status": "pending", "detail": "Converting document to PDF"},
        {"id": "render", "name": "PyMuPDF Page Screenshotting", "status": "pending", "detail": "Rendering 150 DPI page PNGs"},
        {"id": "vision", "name": "Gemma-3-27b-it Vision Filter", "status": "pending", "detail": "Classifying usefulness with google/gemma-3-27b-it"},
        {"id": "extract", "name": "LlamaParse Markdown Extraction", "status": "pending", "detail": "Extracting text and markdown"},
        {"id": "vectorize", "name": "Vector Embedding & MongoDB Atlas Indexing", "status": "pending", "detail": "Embedding chunks and indexing to Atlas"}
    ]

def init_job(job_id: str, filename: str) -> Dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    stages = get_job_stages_template(suffix)
    job = {
        "job_id": job_id,
        "filename": filename,
        "status": "processing",
        "current_stage": stages[0]["name"],
        "progress_percent": 10,
        "stages": stages,
        "chunks_indexed": 0,
        "error": None,
        "error_details": None,
        "started_at": time.time(),
        "completed_at": None
    }
    JOBS[job_id] = job
    return job

def update_job_stage(job_id: str, stage_id: str, stage_status: str, detail: Optional[str] = None, progress: Optional[int] = None):
    job = JOBS.get(job_id)
    if not job:
        return
    
    for stage in job["stages"]:
        if stage["id"] == stage_id:
            stage["status"] = stage_status
            if detail:
                stage["detail"] = detail
            if stage_status == "in_progress":
                job["current_stage"] = stage["name"]
        elif stage_status == "in_progress" and stage["status"] == "pending":
            pass

    if progress is not None:
        job["progress_percent"] = progress

def fail_job(job_id: str, stage_id: str, error_msg: str, exc: Optional[Exception] = None):
    job = JOBS.get(job_id)
    if not job:
        return
    job["status"] = "failed"
    job["error"] = error_msg
    job["error_details"] = traceback.format_exc() if exc else error_msg
    job["completed_at"] = time.time()
    for stage in job["stages"]:
        if stage["id"] == stage_id:
            stage["status"] = "failed"
            stage["detail"] = f"Error: {error_msg}"
            break
    # Persist failure to mongo
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        m_client[MONGO_DB]["ingestion_jobs"].update_one(
            {"job_id": job_id},
            {"$set": job},
            upsert=True
        )
        m_client.close()
    except Exception as me:
        logger.warning(f"Could not persist failed job to mongo: {me}")

def complete_job(job_id: str, chunks_indexed: int):
    job = JOBS.get(job_id)
    if not job:
        return
    job["status"] = "completed"
    job["progress_percent"] = 100
    job["current_stage"] = "Completed Successfully"
    job["chunks_indexed"] = chunks_indexed
    job["completed_at"] = time.time()
    for stage in job["stages"]:
        if stage["status"] != "failed":
            stage["status"] = "completed"
    # Persist to mongo
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        m_client[MONGO_DB]["ingestion_jobs"].update_one(
            {"job_id": job_id},
            {"$set": job},
            upsert=True
        )
        m_client.close()
    except Exception as me:
        logger.warning(f"Could not persist job to mongo: {me}")

def process_file(file_path: Path, client: genai.Client = genai_client, job_id: Optional[str] = None):
    if not job_id:
        job_id = str(uuid.uuid4())
        init_job(job_id, file_path.name)

    logger.info(f"[Job {job_id}] Processing file: {file_path.name}")
    suffix = file_path.suffix.lower()
    records = []

    try:
        if suffix in [".xlsx", ".xls"]:
            # Path B: Excel Files (Zero Screenshots)
            logger.info(f"[Job {job_id}] Running Path B (Excel Tabular) for: {file_path.name}")
            update_job_stage(job_id, "tabular", "in_progress", "Extracting sheets and rows...", progress=40)
            records = parse_excel_file(file_path)
            update_job_stage(job_id, "tabular", "completed", f"Extracted {len(records)} row chunks", progress=60)

        elif suffix in [".pdf", ".docx", ".doc", ".ppt", ".pptx"]:
            # Path A: Visual Documents
            logger.info(f"[Job {job_id}] Running Path A (Visual Conversion & Screenshotting) for: {file_path.name}")
            pdf_dir = OUTPUT_DIR / "pdfs"
            images_dir = OUTPUT_DIR / "images" / file_path.stem
            pdf_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            # 1. Convert to PDF if DOCX/PPT
            update_job_stage(job_id, "convert", "in_progress", "Converting document to PDF...", progress=20)
            try:
                pdf_path = convert_to_pdf(file_path, pdf_dir)
                update_job_stage(job_id, "convert", "completed", f"PDF generated: {pdf_path.name}", progress=35)
            except Exception as conv_err:
                if suffix in [".pptx", ".ppt"]:
                    logger.warning(f"LibreOffice failed ({conv_err}), falling back to python-pptx extraction...")
                    from pipelines.converter import parse_pptx_fallback
                    fallback_slides = parse_pptx_fallback(file_path)
                    if fallback_slides:
                        update_job_stage(job_id, "convert", "completed", f"Extracted {len(fallback_slides)} slides via PPTX parser", progress=40)
                        update_job_stage(job_id, "render", "completed", "Extracted slide structure (fallback mode)", progress=55)
                        update_job_stage(job_id, "vision", "completed", f"Processed {len(fallback_slides)} presentation slides", progress=70)
                        update_job_stage(job_id, "extract", "completed", f"Extracted text from {len(fallback_slides)} slides", progress=85)

                        for s in fallback_slides:
                            records.append({
                                "file_name": file_path.name,
                                "source_type": "visual",
                                "page_number": s["page_number"],
                                "sheet_name": None,
                                "screenshot_url": None,
                                "raw_data": None,
                                "text_content": f"### Document: {file_path.name} | Slide {s['page_number']}\n{s['text']}"
                            })

                        # Proceed directly to vectorization
                        if records:
                            update_job_stage(job_id, "vectorize", "in_progress", f"Generating 768d embeddings for {len(records)} chunks...", progress=90)
                            try:
                                ingest_to_atlas(MONGO_URI, MONGO_DB, records, client)
                                update_job_stage(job_id, "vectorize", "completed", f"Indexed {len(records)} chunks into Atlas", progress=100)
                                complete_job(job_id, len(records))
                                logger.info(f"[Job {job_id}] Ingestion completed successfully.")
                                return
                            except Exception as vect_err:
                                fail_job(job_id, "vectorize", f"Embedding or MongoDB indexing failed: {str(vect_err)}", vect_err)
                                return
                fail_job(job_id, "convert", f"LibreOffice conversion failed: {str(conv_err)}", conv_err)
                return

            # 2. Extract page screenshots
            update_job_stage(job_id, "render", "in_progress", "Rendering 150 DPI page screenshots...", progress=40)
            try:
                screenshots = extract_page_screenshots(pdf_path, images_dir)
                total_pages = len(screenshots)
                update_job_stage(job_id, "render", "completed", f"Rendered {total_pages} page screenshots", progress=50)
            except Exception as rend_err:
                fail_job(job_id, "render", f"PyMuPDF screenshotting failed: {str(rend_err)}", rend_err)
                return

            # 3. Classify with Gemma-3-27b-it Vision Filter
            update_job_stage(job_id, "vision", "in_progress", f"Classifying {total_pages} pages with Gemma-3-27b-it...", progress=55)
            valid_screenshots = []
            for i, sc in enumerate(screenshots, start=1):
                update_job_stage(job_id, "vision", "in_progress", f"Classifying page {i}/{total_pages} with Gemma-3-27b-it...", progress=55 + int((i/total_pages)*15))
                is_valid = classify_page_image(client, sc["image_path"])
                if is_valid:
                    valid_screenshots.append(sc)
                else:
                    logger.info(f"Filtered out page {sc['page_number']} (no useful knowledge).")

            update_job_stage(job_id, "vision", "completed", f"Kept {len(valid_screenshots)}/{total_pages} useful pages", progress=70)

            # 4. Content Extraction (LlamaParse or OCR fallback)
            update_job_stage(job_id, "extract", "in_progress", "Extracting markdown and structured text...", progress=75)
            for sc in valid_screenshots:
                markdown_content = parse_page_with_llamaparse(sc["image_path"], LLAMA_CLOUD_API_KEY)
                if not markdown_content:
                    extracted_text = sc.get("text", "")
                    if len(extracted_text) < 25:
                        extracted_text = extract_image_description(client, sc["image_path"])
                    markdown_content = extracted_text

                screenshot_url = f"/static/images/{file_path.stem}/{sc['image_path'].name}"
                content = f"### Document: {file_path.name} | Page {sc['page_number']}\n{markdown_content}"

                records.append({
                    "file_name": file_path.name,
                    "source_type": "visual",
                    "page_number": sc["page_number"],
                    "sheet_name": None,
                    "screenshot_url": screenshot_url,
                    "raw_data": None,
                    "text_content": content
                })
            update_job_stage(job_id, "extract", "completed", f"Extracted {len(records)} page contents", progress=85)

        else:
            fail_job(job_id, "upload", f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .ppt, .pptx, .xlsx, .xls")
            return

        if records:
            update_job_stage(job_id, "vectorize", "in_progress", f"Generating 768d embeddings for {len(records)} chunks...", progress=90)
            try:
                ingest_to_atlas(MONGO_URI, MONGO_DB, records, client)
                update_job_stage(job_id, "vectorize", "completed", f"Indexed {len(records)} chunks into Atlas", progress=100)
                complete_job(job_id, len(records))
                logger.info(f"[Job {job_id}] Ingestion completed successfully.")
            except Exception as vect_err:
                fail_job(job_id, "vectorize", f"Embedding or MongoDB indexing failed: {str(vect_err)}", vect_err)
        else:
            fail_job(job_id, "extract", "No valid content or pages were extracted from document.")

    except Exception as e:
        logger.error(f"[Job {job_id}] Pipeline exception: {e}", exc_info=True)
        fail_job(job_id, "upload", f"Unexpected ingestion failure: {str(e)}", e)

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        m_client.admin.command("ping")
        coll = m_client[MONGO_DB]["vectors"]
        count = coll.count_documents({})
        m_client.close()
        return {
            "status": "healthy",
            "service": "ingestion-service",
            "database_connected": True,
            "vectors_ingested_count": count
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "ingestion-service",
            "database_connected": False,
            "error": str(e)
        }

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    init_job(job_id, file.filename)
    
    file_path = DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        fail_job(job_id, "upload", f"Failed to save uploaded file: {str(e)}", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Run tracked pipeline in background
    background_tasks.add_task(process_file, file_path, genai_client, job_id)
    return {
        "job_id": job_id,
        "file_name": file.filename,
        "status": "processing",
        "message": f"'{file.filename}' uploaded successfully. Tracking pipeline execution."
    }

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        # Check MongoDB if not in memory
        try:
            m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            doc = m_client[MONGO_DB]["ingestion_jobs"].find_one({"job_id": job_id}, {"_id": 0})
            m_client.close()
            if doc:
                return doc
        except Exception:
            pass
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@app.get("/jobs")
def list_jobs():
    # Return jobs sorted by started_at desc
    job_list = list(JOBS.values())
    job_list.sort(key=lambda x: x.get("started_at", 0), reverse=True)
    return {"jobs": job_list[:25]}

@app.get("/documents")
def get_documents():
    try:
        m_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        coll = m_client[MONGO_DB]["vectors"]
        pipeline = [
            {
                "$group": {
                    "_id": "$file_name",
                    "source_type": {"$first": "$source_type"},
                    "chunks": {"$sum": 1},
                    "screenshot_url": {"$first": "$screenshot_url"}
                }
            },
            {
                "$project": {
                    "file_name": "$_id",
                    "source_type": 1,
                    "chunks": 1,
                    "screenshot_url": 1,
                    "_id": 0
                }
            },
            {"$sort": {"file_name": 1}}
        ]
        docs = list(coll.aggregate(pipeline))
        m_client.close()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        return {"documents": [], "error": str(e)}

# Background watcher thread for files dropped into data_drop directory
def background_watcher():
    processed_files = set()
    logger.info(f"Ingestion directory watcher active on {DATA_DIR}")
    while True:
        try:
            files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
            for file in files:
                file_key = f"{file.name}_{file.stat().st_mtime}"
                if file_key not in processed_files:
                    try:
                        watcher_job_id = str(uuid.uuid4())
                        init_job(watcher_job_id, file.name)
                        process_file(file, genai_client, watcher_job_id)
                        processed_files.add(file_key)
                    except Exception as e:
                        logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Watcher loop exception: {e}")
        time.sleep(3)

# Start watcher thread when service starts
watcher_thread = threading.Thread(target=background_watcher, daemon=True)
watcher_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
