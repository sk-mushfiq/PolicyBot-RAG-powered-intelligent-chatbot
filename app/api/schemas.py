"""
app/api/schemas.py
───────────────────
Pydantic models for all API request and response bodies.
FastAPI uses these for automatic validation and OpenAPI docs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────
# Ingest Endpoint
# ─────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response returned after successful document ingestion."""
    message: str
    filename: str
    chunks_created: int

    model_config = {"json_schema_extra": {
        "example": {
            "message": "Document ingested successfully.",
            "filename": "leave_policy.pdf",
            "chunks_created": 42,
        }
    }}


class IngestDirectoryResponse(BaseModel):
    """Response returned after ingesting an entire directory."""
    message: str
    directory: str
    total_chunks: int
    files_processed: int


# ─────────────────────────────────────────────────────
# Chat Endpoint
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question about HR policies.",
    )
    session_id: str = Field(
        default="default",
        description="Session identifier for conversation continuity.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "How many annual leave days do I get?",
            "session_id": "user_123",
        }
    }}


class SourceCitation(BaseModel):
    """A single source document citation."""
    source: str = Field(description="Document filename.")
    page: int = Field(description="Page number (0 for DOCX files).")


class ChatResponse(BaseModel):
    """Response returned by the /chat endpoint."""
    answer: str = Field(description="LLM-generated answer.")
    sources: List[SourceCitation] = Field(
        description="Source documents used to generate the answer."
    )
    session_id: str = Field(description="Echo of the session identifier.")

    model_config = {"json_schema_extra": {
        "example": {
            "answer": "You are entitled to 20 days of annual leave per year.",
            "sources": [{"source": "leave_policy.pdf", "page": 3}],
            "session_id": "user_123",
        }
    }}


# ─────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: str
    app_name: str
    version: str
    vector_store_ready: bool
