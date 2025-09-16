import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from functions import upload_and_index, semantic_search, load_index_from_db
from contextlib import asynccontextmanager
from models import SearchRequest, SearchResult
from typing import Dict,List
# --------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------
# FastAPI Application with Lifespan
# --------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Runs on startup and shutdown.
    """
    # --- Runs on startup ---
    load_index_from_db()
    yield
    # --- Runs on shutdown ---
    logging.info("Shutting down app...")


app = FastAPI(
    title="QueryNest Backend",
    description="Semantic PDF search using FastAPI, FAISS, and Supabase.",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------------
# API Routes
# --------------------------------------------------------

@app.post("/upload_pdf", tags=["PDF Management"])
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload a PDF file, extract text, create embeddings,
    store in Supabase, and index in FAISS.

    Returns:
        dict: Message about upload status.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported."
        )
    return upload_and_index(file)


@app.post("/search", tags=["Semantic Search"], response_model=List[SearchResult])
async def search_files(request: SearchRequest) -> List[SearchResult]:
    """
    Perform semantic search on uploaded PDFs.

    Args:
        request (SearchRequest): Search query and number of results.

    Returns:
        List[SearchResult]: List of matched PDFs with snippets.
    """
    results = semantic_search(request)
    if not results:
        raise HTTPException(status_code=404, detail="No PDFs uploaded yet.")
    return results


@app.get("/", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API is running.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}