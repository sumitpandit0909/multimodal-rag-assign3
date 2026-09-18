import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent.parent

# Load .env from root and service directory
load_dotenv()
load_dotenv(ROOT_DIR / ".env")

# Mongo Atlas Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "multimodal_rag")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Cloudflare R2 Credentials
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")

# Working Directories
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data_drop"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "processed_output"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Google GenAI Client
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
