"""
tests/test_vector_store.py
───────────────────────────
Tests for VectorStoreManager using an in-memory ChromaDB instance
(no disk writes) to keep tests fast and isolated.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def make_documents(n: int = 3) -> list[Document]:
    """Create n synthetic Document chunks for testing."""
    return [
        Document(
            page_content=f"Employees get {i * 5} days of leave per year.",
            metadata={"source": "leave_policy.pdf", "page": i},
        )
        for i in range(1, n + 1)
    ]


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture
def vector_store(tmp_path):
    """
    Return a VectorStoreManager using a temp ChromaDB dir.
    Uses HuggingFace embeddings to avoid needing an OpenAI key in tests.
    """
    with patch("app.services.vector_store.get_settings") as mock_settings:
        settings = MagicMock()
        settings.chroma_collection_name = "test_collection"
        settings.chroma_persist_dir = str(tmp_path / "chroma_test")
        settings.embedding_provider = "huggingface"
        settings.huggingface_model = "all-MiniLM-L6-v2"
        settings.top_k_results = 2
        mock_settings.return_value = settings

        from app.services.vector_store import VectorStoreManager
        store = VectorStoreManager(
            collection_name="test_collection",
            persist_dir=str(tmp_path / "chroma_test"),
        )
        yield store


# ─────────────────────────────────────────────────────
# Tests — Add Documents
# ─────────────────────────────────────────────────────

class TestAddDocuments:
    def test_empty_list_raises_value_error(self, vector_store):
        """Adding an empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty document list"):
            vector_store.add_documents([])

    def test_returns_count_of_added_documents(self, vector_store):
        """Return value equals the number of documents passed in."""
        docs = make_documents(3)
        count = vector_store.add_documents(docs)
        assert count == 3

    def test_single_document_stored(self, vector_store):
        """A single document can be stored without error."""
        docs = make_documents(1)
        count = vector_store.add_documents(docs)
        assert count == 1


# ─────────────────────────────────────────────────────
# Tests — Similarity Search
# ─────────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_search_returns_list(self, vector_store):
        """Similarity search always returns a list."""
        vector_store.add_documents(make_documents(3))
        results = vector_store.similarity_search("annual leave days")
        assert isinstance(results, list)

    def test_search_respects_k_parameter(self, vector_store):
        """Results count does not exceed k."""
        vector_store.add_documents(make_documents(5))
        results = vector_store.similarity_search("leave policy", k=2)
        assert len(results) <= 2

    def test_search_results_are_documents(self, vector_store):
        """Each result is a LangChain Document instance."""
        vector_store.add_documents(make_documents(3))
        results = vector_store.similarity_search("leave")
        for doc in results:
            assert isinstance(doc, Document)

    def test_search_result_has_page_content(self, vector_store):
        """Each result has non-empty page_content."""
        vector_store.add_documents(make_documents(3))
        results = vector_store.similarity_search("employees leave")
        for doc in results:
            assert doc.page_content.strip() != ""

    def test_search_result_has_metadata(self, vector_store):
        """Each result preserves source metadata."""
        vector_store.add_documents(make_documents(3))
        results = vector_store.similarity_search("leave")
        for doc in results:
            assert "source" in doc.metadata


# ─────────────────────────────────────────────────────
# Tests — Collection State
# ─────────────────────────────────────────────────────

class TestCollectionState:
    def test_empty_collection_returns_true(self, vector_store):
        """collection_is_empty() returns True when nothing is stored."""
        assert vector_store.collection_is_empty() is True

    def test_after_add_collection_not_empty(self, vector_store):
        """collection_is_empty() returns False after adding documents."""
        vector_store.add_documents(make_documents(2))
        assert vector_store.collection_is_empty() is False

    def test_get_retriever_returns_retriever(self, vector_store):
        """get_retriever() returns a LangChain VectorStoreRetriever."""
        vector_store.add_documents(make_documents(2))
        retriever = vector_store.get_retriever(k=1)
        assert retriever is not None
        assert hasattr(retriever, "invoke") or hasattr(retriever, "get_relevant_documents")
