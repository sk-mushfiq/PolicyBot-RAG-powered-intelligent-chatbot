"""
tests/test_api.py
──────────────────
Integration tests for FastAPI routes using TestClient.
All external services (vector store, RAG chain) are mocked.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with all service dependencies mocked."""

    # Patch module-level singletons in both route files
    with patch("app.api.routes.ingest._processor") as mock_proc, \
         patch("app.api.routes.ingest._vector_store") as mock_vs_ingest, \
         patch("app.api.routes.chat._rag_chain") as mock_chain, \
         patch("app.api.routes.chat._memory_manager") as mock_memory:

        # --- Ingest mocks ---
        mock_proc.load_and_split.return_value = [
            MagicMock(metadata={"source": "test.pdf", "page": 0})
            for _ in range(5)
        ]
        mock_vs_ingest.add_documents.return_value = 5

        # --- Chat mocks ---
        mock_chain.query.return_value = {
            "answer": "You get 20 days annual leave per year.",
            "sources": [{"source": "leave_policy.pdf", "page": 3}],
            "session_id": "test_session",
        }

        # --- Memory mocks ---
        mock_memory.get_history.return_value = []

        from app.main import create_app
        app = create_app()
        yield TestClient(app)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal PDF bytes for upload testing."""
    # Minimal valid PDF structure
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
    )


# ─────────────────────────────────────────────────────
# Tests — Health Check
# ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /health returns 200 OK."""
        with patch("app.main.VectorStoreManager") as mock_vs:
            mock_vs.return_value.collection_is_empty.return_value = False
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_key(self, client):
        """Health response contains a 'status' key."""
        with patch("app.main.VectorStoreManager") as mock_vs:
            mock_vs.return_value.collection_is_empty.return_value = True
            response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_root_endpoint_returns_200(self, client):
        """GET / returns 200."""
        response = client.get("/")
        assert response.status_code == 200


# ─────────────────────────────────────────────────────
# Tests — Ingest Endpoint
# ─────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_valid_pdf_returns_201(self, client, sample_pdf_bytes):
        """Valid PDF upload returns 201 Created."""
        response = client.post(
            "/ingest",
            files={"file": ("leave_policy.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201

    def test_ingest_response_schema(self, client, sample_pdf_bytes):
        """Response contains message, filename, and chunks_created."""
        response = client.post(
            "/ingest",
            files={"file": ("policy.pdf", sample_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert "message" in data
        assert "filename" in data
        assert "chunks_created" in data

    def test_ingest_unsupported_file_type_returns_422(self, client):
        """Uploading a .txt file returns 422 Unprocessable Entity."""
        response = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"Some text content", "text/plain")},
        )
        assert response.status_code == 422

    def test_ingest_csv_file_returns_422(self, client):
        """Uploading a .csv file returns 422."""
        response = client.post(
            "/ingest",
            files={"file": ("data.csv", b"name,value\nfoo,bar", "text/csv")},
        )
        assert response.status_code == 422

    def test_ingest_returns_correct_filename(self, client, sample_pdf_bytes):
        """Response filename matches the uploaded file's name."""
        response = client.post(
            "/ingest",
            files={"file": ("my_policy.pdf", sample_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["filename"] == "my_policy.pdf"


# ─────────────────────────────────────────────────────
# Tests — Chat Endpoint
# ─────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_chat_valid_request_returns_200(self, client):
        """Valid chat request returns 200."""
        response = client.post(
            "/chat",
            json={"question": "How many leave days?", "session_id": "s1"},
        )
        assert response.status_code == 200

    def test_chat_response_has_answer(self, client):
        """Chat response contains an 'answer' key."""
        response = client.post(
            "/chat",
            json={"question": "What is the leave policy?"},
        )
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_chat_response_has_sources(self, client):
        """Chat response contains a 'sources' list."""
        response = client.post(
            "/chat",
            json={"question": "Leave policy?", "session_id": "s2"},
        )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_chat_response_has_session_id(self, client):
        """Chat response echoes the session_id."""
        response = client.post(
            "/chat",
            json={"question": "Any question", "session_id": "my_session"},
        )
        data = response.json()
        assert "session_id" in data

    def test_chat_empty_question_returns_422(self, client):
        """Empty question string is rejected with 422."""
        response = client.post(
            "/chat",
            json={"question": "", "session_id": "s3"},
        )
        assert response.status_code == 422

    def test_chat_missing_question_field_returns_422(self, client):
        """Missing 'question' field returns 422."""
        response = client.post(
            "/chat",
            json={"session_id": "s4"},
        )
        assert response.status_code == 422

    def test_chat_when_store_empty_returns_503(self, client):
        """503 is returned when no documents have been ingested."""
        with patch("app.api.routes.chat._rag_chain") as mock_chain:
            mock_chain.query.side_effect = RuntimeError(
                "No documents found in the vector store."
            )
            response = client.post(
                "/chat",
                json={"question": "Any question?", "session_id": "s5"},
            )
        assert response.status_code == 503


# ─────────────────────────────────────────────────────
# Tests — Session Management
# ─────────────────────────────────────────────────────

class TestSessionManagement:
    def test_clear_session_returns_200(self, client):
        """DELETE /chat/session/{id} returns 200."""
        response = client.delete("/chat/session/test_session")
        assert response.status_code == 200

    def test_clear_session_response_message(self, client):
        """Clear session response contains a confirmation message."""
        response = client.delete("/chat/session/abc123")
        data = response.json()
        assert "message" in data

    def test_get_history_returns_200(self, client):
        """GET /chat/history/{id} returns 200."""
        response = client.get("/chat/history/test_session")
        assert response.status_code == 200

    def test_get_history_has_required_keys(self, client):
        """History response contains session_id, message_count, history."""
        response = client.get("/chat/history/test_session")
        data = response.json()
        assert "session_id" in data
        assert "message_count" in data
        assert "history" in data
