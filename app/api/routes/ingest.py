"""
app/api/routes/ingest.py
─────────────────────────
POST /ingest — upload and process a single HR document.
POST /ingest/directory — ingest all documents from a server-side path.
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.schemas import IngestResponse, IngestDirectoryResponse
from app.core.logger import get_logger
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreManager

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Module-level singletons — shared across requests
_processor = DocumentProcessor()
_vector_store = VectorStoreManager()


@router.post("", response_model=IngestResponse, status_code=201)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF or DOCX file, chunk it, embed it, and store in ChromaDB.

    - **file**: PDF or DOCX file to ingest.

    Returns the number of chunks created.
    """
    # Validate extension before doing any work
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Upload PDF or DOCX.",
        )

    # Write upload to a temp file (UploadFile is a stream)
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        logger.info(f"Ingesting uploaded file: {file.filename}")
        chunks = _processor.load_and_split(tmp_path)

        # Override source metadata with the original filename
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        count = _vector_store.add_documents(chunks)
        logger.info(
            f"Ingestion complete | file='{file.filename}' chunks={count}"
        )

        return IngestResponse(
            message="Document ingested successfully.",
            filename=file.filename,
            chunks_created=count,
        )

    except Exception as exc:
        logger.error(f"Ingestion failed for '{file.filename}': {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        os.unlink(tmp_path)   # Always clean up temp file


@router.post("/directory", response_model=IngestDirectoryResponse, status_code=201)
async def ingest_directory(directory: str) -> IngestDirectoryResponse:
    """
    Ingest all PDF/DOCX files from a server-side directory path.

    - **directory**: Absolute path to a folder on the server.

    Useful for bulk loading documents at startup.
    """
    if not Path(directory).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {directory}",
        )

    try:
        chunks = _processor.load_directory(directory)
        if not chunks:
            return IngestDirectoryResponse(
                message="No supported files found in directory.",
                directory=directory,
                total_chunks=0,
                files_processed=0,
            )

        count = _vector_store.add_documents(chunks)

        # Count unique source files
        unique_sources = len({c.metadata.get("source") for c in chunks})

        return IngestDirectoryResponse(
            message="Directory ingested successfully.",
            directory=directory,
            total_chunks=count,
            files_processed=unique_sources,
        )

    except Exception as exc:
        logger.error(f"Directory ingestion failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
