import logging
import io
import os
import json
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase import create_client
from PyPDF2 import PdfReader
from fastapi import UploadFile
from models import SearchRequest, SearchResult


# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FAISS & Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)
dimension = 384
index = faiss.IndexFlatL2(dimension)

# In-memory storage for metadata
pdf_texts: List[Dict[str, str]] = []


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of a file's content.
    Used for duplicate detection before uploading.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract all text from a given PDF file.
    Returns concatenated text of all pages.
    """
    reader = PdfReader(file.file)
    text: str = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# --------------------------------------------------------
# Uploading & Indexing Functions
# --------------------------------------------------------

def upload_and_index(file: UploadFile) -> Dict[str, str]:
    """
    Upload a PDF file to Supabase storage,
    extract text, generate embeddings, update FAISS,
    and store metadata in Supabase database.
    """
    # Read file bytes
    file_bytes: bytes = file.file.read()
    file.file.seek(0)  # reset pointer

    # 1. Compute hash for duplicate detection
    file_hash: str = compute_file_hash(file_bytes)
    existing = supabase.table("documents").select("*").eq("file_hash", file_hash).execute()
    if existing.data:
        return {"message": f"{file.filename} already uploaded."}

    # 2. Upload to Supabase Storage
    file_path: str = f"{file.filename}"
    supabase.storage.from_(BUCKET_NAME).upload(
        path=file_path,
        file=file_bytes,
        file_options={
            "upsert": "false",
            "content-type": "application/pdf"
        }
    )

    # 3. Extract text from PDF
    text: str = extract_text_from_pdf(file)
    if not text.strip():
        return {"message": "No extractable text found."}

    # 4. Generate embeddings and update FAISS
    embedding: np.ndarray = model.encode([text])
    index.add(embedding)
    pdf_texts.append({"filename": file.filename, "content": text})

    # 5. Insert metadata into Supabase DB
    supabase.table("documents").insert({
        "filename": file.filename,
        "file_path": file_path,
        "content": text,
        "embedding": embedding[0].tolist(),
        "file_hash": file_hash
    }).execute()

    logging.info(f"Uploaded and indexed PDF: {file.filename}")
    return {"message": f"File {file.filename} uploaded and indexed."}


# --------------------------------------------------------
# Loading Data into FAISS
# --------------------------------------------------------

def load_index_from_db() -> None:
    """
    Load stored embeddings and document metadata
    from Supabase database into FAISS and memory.
    """
    global index, pdf_texts

    # Reset FAISS and local memory
    pdf_texts.clear()
    index.reset()

    # Fetch all stored documents
    response = supabase.table("documents").select("filename, content, embedding").execute()
    docs: List[Dict[str, Any]] = response.data or []

    if not docs:
        logging.info("No documents found in Supabase DB.")
        return

    embeddings: List[List[float]] = []
    for doc in docs:
        pdf_texts.append({"filename": doc["filename"], "content": doc["content"]})
        embedding = doc["embedding"]
        # Ensure embedding is a list of floats
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        embeddings.append(embedding)

    # Add embeddings into FAISS
    if embeddings:
        embeddings_np: np.ndarray = np.array(embeddings).astype("float32")
        index.add(embeddings_np)

    logging.info(f"Loaded {len(docs)} documents into FAISS.")


# --------------------------------------------------------
# Semantic Search
# --------------------------------------------------------

def semantic_search(request: SearchRequest) -> List[SearchResult]:
    """
    Perform semantic search on indexed documents.
    Returns a list of SearchResult with filename and text snippet.
    """
    if len(pdf_texts) == 0:
        return []

    # Encode query and search in FAISS
    query_embedding: np.ndarray = model.encode([request.query])
    D, I = index.search(query_embedding, request.top_k)

    results: List[SearchResult] = []
    for idx in I[0]:
        if idx == -1 or idx >= len(pdf_texts):
            continue
        file_info = pdf_texts[idx]
        snippet: str = file_info["content"][:200] + "..."
        url = SUPABASE_URL + "/storage/v1/object/public/" + BUCKET_NAME + "/" + file_info["filename"]
        
        results.append(SearchResult(filename=file_info["filename"], snippet=snippet, url=url))


    return results
