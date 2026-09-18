# Enterprise Multimodal RAG & Microservices Architecture Guide

> **Status:** Production-Ready with LangGraph Agentic RAG  
> **Repository:** `assignment3`  
> **Last Updated:** 2026-09-18

---

## 1. High-Level Architecture Overview

The system is built as a **fully decoupled microservices architecture** where independent services communicate exclusively through a shared **MongoDB Atlas** database and well-defined HTTP REST endpoints.

```
                                  +---------------------------------------+
                                  |         React 19 + TypeScript         |
                                  |         Frontend (Port 5173)          |
                                  +-------------------+-------------------+
                                                      |
                          +---------------------------+---------------------------+
                          | HTTP (Live Polling / REST)                            | HTTP (REST)
                          v                                                       v
        +-----------------------------------+                   +-----------------------------------+
        |        Ingestion Service          |                   |           Chat Service            |
        |            (Port 8001)            |                   |            (Port 8000)            |
        | - LibreOffice / python-pptx       |                   | - Fast, Async Motor Client        |
        | - PyMuPDF (150 DPI Screenshots)   |                   | - LangGraph Agentic StateGraph    |
        | - Gemma-3-27b-it Vision Filter    |                   |   * Retrieve (MongoDB Atlas)      |
        | - LlamaParse Markdown Extractor   |                   |   * Document Relevance Grading    |
        | - Excel Tabular Chunking (No PNG) |                   |   * Self-Corrective Query Rewrite |
        | - Gemini-Embedding-001 (768d)     |                   |   * Grounded Synthesis [1], [2]   |
        | - Mongo Atlas Vector Ingestion    |                   | - Multi-Turn Session Memory       |
        +-----------------+-----------------+                   | - Native LangSmith Tracing        |
                          |                                     +-----------------+-----------------+
                          |                                                       |
                          +---------------------------+---------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |    MongoDB Atlas Cluster  |
                                        |  Database: multimodal_rag |
                                        |                           |
                                        | - vectors (768d Atlas IX) |
                                        | - conversations           |
                                        | - ingestion_jobs          |
                                        +---------------------------+
```

---

## 2. Decoupling & Microservice Boundaries

1. **Ingestion Service (`port 8001`)**:
   - Solely responsible for document parsing, PDF conversion, image rendering, vision filtering, markdown extraction, vector embedding, and storing records in MongoDB Atlas.
   - Operates independently via file uploads (`POST /upload`), background job tracking (`GET /jobs/{job_id}`), and a directory watcher on `data_drop/`.
   - Has **zero** dependency on `chat-service`.

2. **Chat Service (`port 8000`)**:
   - Built on a **LangGraph Self-Corrective Agentic RAG** state machine.
   - Handles conversational queries, multi-turn session memory, Atlas vector search, document relevance grading, query rewriting, and answer generation with interactive citation badges (`[1]`, `[2]`).
   - Has **zero** dependency on `ingestion-service`.

3. **Frontend (`port 5173` -> Docker Nginx on `:80`)**:
   - Consumes `chat-service` (`http://localhost:8000`) for conversation and document counts.
   - Consumes `ingestion-service` (`http://localhost:8001`) for file uploads, live stage progress polling, and static page screenshot assets (`/static/images/...`).

4. **Shared Database (`multimodal_rag` on MongoDB Atlas)**:
   - `vectors`: Stores document chunks, 768-dimensional embeddings, source types (`visual` vs `tabular`), page numbers, sheet names, screenshot URLs, and raw sanitized table rows.
   - `conversations`: Stores multi-turn session chat history and verified source nodes.
   - `ingestion_jobs`: Stores live state, progress percentages, active stages, and failure logs for background jobs.

---

## 3. Technology Stack & AI Models

| Component / Task | Technology / Model Used | Purpose / Implementation Details |
| :--- | :--- | :--- |
| **Agent Orchestration** | `LangGraph` (`langgraph`, `langchain-core`) | Self-Corrective Agentic RAG state machine with document grading, query rewriting, and conditional loops. |
| **Vision Classification** | `google/gemma-3-27b-it` (OpenRouter) | Evaluates page screenshots to filter out blank/useless slides (`YES`/`NO`). Fallback: `gemini-2.0-flash`. |
| **RAG Answer Generation** | `google/gemma-3-27b-it` (OpenRouter) | Synthesizes grounded answers citing verified sources (`[1]`, `[2]`). Fallback: `gemini-2.0-flash`. |
| **Vector Embeddings** | `gemini-embedding-001` (Google GenAI) | Generates 768-dimensional dense vector embeddings with exponential backoff retries. |
| **PDF Conversion** | `libreoffice-nogui` + `default-jre-headless` | Headless batch conversion for DOCX, PPT, PPTX to PDF. |
| **Presentation Fallback** | `python-pptx` | Native Python parser extracting slides, shapes, text, and tables if headless graphics fail. |
| **Screenshot Rendering** | `PyMuPDF` (`fitz`) | Renders crisp 150 DPI page PNG screenshots for visual inspection. |
| **Structured OCR/Markdown** | `LlamaParse` (`llama-parse`) | Cloud extraction of rich markdown, complex tables, and diagrams from document pages. |
| **Tabular Data Processing** | `pandas` + `openpyxl` | Chunks sheets into 20-row batches, converts to markdown, and sanitizes BSON types (**zero screenshots**). |
| **Database & Vector Search** | `MongoDB Atlas` (`pymongo` & `motor`) | 768-dimensional `$vectorSearch` index, document storage, and session memory. |
| **Tracing & Observability** | `LangSmith` (`langsmith`) | First-party hierarchical graph tracing and `@traceable` instrumentation. |
| **Frontend UI** | `React 19` + `TypeScript` + `Vite` | Minimalist dark theme, real-time stage stepper, `react-markdown` parser, zoom/pan visual inspector. |

---

## 4. LangGraph Agentic RAG Architecture (`chat-service`)

The `chat-service` orchestrates query answering using a compiled LangGraph `StateGraph`:

```
   [START]
      │
      ▼
 [retrieve]
      │
      ▼
[grade_documents]
      │
      ├── (is_relevant == True)  ──▶ [generate]   ──▶ [END]
      └── (is_relevant == False) ──▶ [no_sources] ──▶ [END]
```

### LangGraph State Schema (`AgentState`)
```python
class AgentState(TypedDict):
    query: str                  # User inquiry
    history: List[Dict]         # Multi-turn chat session history
    retrieved_sources: List[Dict]# Vector chunks from MongoDB Atlas
    documents_relevant: bool    # Grader output flag
    answer: str                 # Final synthesized answer
    source_nodes: List[Dict]    # Formatted source citations for frontend
```

### Node Descriptions
1. **`retrieve`**: Calls `search_vector_store` (cached 768d query embedding + MongoDB Atlas `$vectorSearch`).
2. **`grade_documents`**: Strict semantic evaluation using `google/gemini-2.5-flash-lite` (~200ms) to ensure zero false-positive citations.
3. **`generate`**: Synthesizes the final answer citing verified sources (`[1]`, `[2]`) using `google/gemma-3-27b-it` (fallback to `gemini-2.0-flash`). Only referenced citations are returned to the frontend.
4. **`no_sources`**: Formulates a polite, factual response when no relevant documents exist. Returns `source_nodes: []` (0 images).

---

## 5. File-by-File Breakdown

### Root Directory
* **`docker-compose.yml`**: Orchestrates `chat-service` (8000), `ingestion-service` (8001), and `frontend` (5173).
* **`.env` / `.env.example`**: Defines secrets (`MONGO_URI`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `LLAMA_CLOUD_API_KEY`, `LANGCHAIN_*`).
* **`data_drop/`**: Watched folder for automatic file ingestion.
* **`processed_output/`**: Stores generated PDFs and 150 DPI page PNG screenshots.

### Ingestion Service (`ingestion-services/`)
* **`Dockerfile`**: Debian Python 3.11 container with `libreoffice-nogui`, `default-jre-headless`, and PyMuPDF dependencies.
* **`requirements.txt`**: Core dependencies (`fastapi`, `pymongo`, `google-genai`, `openai`, `langsmith`, `llama-parse`, `pymupdf`, `pandas`, `openpyxl`, `python-pptx`, `boto3`).
* **`src/main.py`**: Slim FastAPI application factory with lifespan (startup stale job sweep + background directory watcher) and router registration.
* **`src/core/config.py`**: Centralized environment variable management, path definitions, and GenAI client.
* **`src/models/schemas.py`**: Pydantic data schemas (`JobStage`, `JobStatus`, `UploadResponse`, `DocumentListResponse`, `HealthResponse`).
* **`src/services/job_manager.py`**: Thread-safe job state tracker with live MongoDB Atlas synchronization and stale job sweeper.
* **`src/services/pipeline_orchestrator.py`**: High-performance orchestrator running page vision classification, LlamaParse extraction, and R2 uploads in parallel (`ThreadPoolExecutor` 4 workers).
* **`src/services/watcher.py`**: Background directory watcher for `data_drop/` with file write completion verification.
* **`src/routes/ingestion_routes.py`**: Modular FastAPI `APIRouter` exposing `/upload`, `/jobs/{job_id}`, `/jobs`, `/documents`, and `/health`.
* **`src/pipelines/converter.py`**: Headless LibreOffice conversion with unique profile isolation (`-env:UserInstallation`) + `python-pptx` presentation parser fallback.
* **`src/pipelines/visual_pipeline.py`**: PyMuPDF 150 DPI PNG page screenshotting.
* **`src/pipelines/vision_filter.py`**: Gemma-3-27b-it OpenRouter vision filter (`YES`/`NO`).
* **`src/pipelines/llama_parser.py`**: Layout-aware markdown parsing with LlamaParse.
* **`src/pipelines/excel_pipeline.py`**: Dynamic token-aware row chunking based on column width, markdown conversion, BSON sanitization (**no screenshots**).
* **`src/storage/vector_store.py`**: Batch 768d embedding generation with automatic file de-duplication in MongoDB Atlas.
* **`src/storage/r2_storage.py`**: Cloudflare R2 S3-compatible cloud object storage manager for page screenshots and converted PDFs with automatic fallback to local `/static` storage.

### Chat Service (`chat-service/`)
* **`Dockerfile`**: Lightweight Python 3.11 container running Uvicorn on port 8000.
* **`requirements.txt`**: Added `langgraph>=0.2.20` and `langchain-core>=0.3.0`.
* **`src/main.py`**: Clean FastAPI application factory, CORS setup, static screenshot mount, and router registration.
* **`src/models/schemas.py`**: Pydantic data models (`ChatRequest`, `ChatResponse`, `SourceNode`, `DocumentListResponse`, `HealthResponse`).
* **`src/db/connection.py`**: Singleton connections for MongoDB Atlas (`AsyncIOMotorClient`) and Google GenAI.
* **`src/db/memory.py`**: Asynchronous multi-turn conversation memory (`MongoChatMemory`).
* **`src/agent/state.py`**: Typed state dictionary (`AgentState`) for the LangGraph state machine.
* **`src/agent/llm.py`**: OpenRouter Gemma-3-27b-it completions, Gemini fallback logic, and system prompts.
* **`src/agent/tools.py`**: MongoDB Atlas 768d `$vectorSearch` retriever tool.
* **`src/agent/nodes.py`**: Granular LangGraph nodes (`retrieve`, `grade_documents`, `rewrite_query`, `generate`, `no_sources`) and conditional routing.
* **`src/agent/graph.py`**: StateGraph assembly, transition wiring, graph compilation, and `run_agentic_rag`.
* **`src/routes/chat_routes.py`**: Modular FastAPI `APIRouter` exposing `/chat`, `/documents`, and `/health`.

### Frontend (`frontend/`)
* **`src/App.tsx`**: Chat container, modal manager, and health monitor.
* **`src/components/IngestionModal.tsx`**: Live stage stepper polling `/jobs/{job_id}` with error alerts.
* **`src/components/MessageBubble.tsx`**: ReactMarkdown renderer with AST citation badge interceptors (`[1]`, `[2]`).
* **`src/components/VisualModal.tsx`**: High-resolution pan-and-zoom page screenshot viewer.
* **`src/components/ExcelDataCard.tsx`**: Interactive tabular sheet data inspector.
* **`src/types.ts`**: Unified TypeScript data contracts.
