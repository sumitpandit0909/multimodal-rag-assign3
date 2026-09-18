import time
import uuid
import logging
from pathlib import Path
from core.config import DATA_DIR, genai_client
from services.job_manager import job_manager
from services.pipeline_orchestrator import process_file

logger = logging.getLogger(__name__)

def is_file_ready(file_path: Path, wait_seconds: float = 0.5) -> bool:
    """Verifies that a file has finished writing and is not partially copied."""
    try:
        size_1 = file_path.stat().st_size
        time.sleep(wait_seconds)
        size_2 = file_path.stat().st_size
        return size_1 == size_2 and size_1 > 0
    except Exception:
        return False

def start_background_watcher():
    """Background watcher thread for files dropped into DATA_DIR."""
    processed_files = set()
    logger.info(f"[Watcher] Directory watcher started on {DATA_DIR}")
    
    while True:
        try:
            files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
            for file in files:
                file_key = f"{file.name}_{file.stat().st_mtime}"
                if file_key not in processed_files:
                    # Check write completion before processing
                    if not is_file_ready(file):
                        continue

                    try:
                        watcher_job_id = str(uuid.uuid4())
                        job_manager.init_job(watcher_job_id, file.name)
                        logger.info(f"[Watcher] Detected new/modified file: {file.name}. Starting job {watcher_job_id}")
                        process_file(file, genai_client, watcher_job_id)
                        processed_files.add(file_key)
                    except Exception as e:
                        logger.error(f"[Watcher] Ingestion failed for {file.name}: {e}")
        except Exception as loop_err:
            logger.error(f"[Watcher] Error in watcher loop: {loop_err}")
        
        time.sleep(5)
