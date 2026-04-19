"""
tests/test_rag_chain.py
────────────────────────
Tests for RAGChain — mocking LLM and vector store calls
so tests run without network access or API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

def make_mock_source_docs():
    return [
        Document(
            page_content="Employees are entitled to 20 days annual leave.",
            metadata={"source": "leave_policy.pdf", "page": 1},
        ),
        Document(
            page_content="Leave must be approved by the line manager.",
            metadata={"source": "leave_policy.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.collection_is_empty.return_value = False
    store.get_retriever.return_value = MagicMock()
    return store


@pytest.fixture
def mock_memory_manager():
    manager = MagicMock()
    manager.get_memory.return_value = MagicMock()
    return manager


@pytest.fixture
def rag_chain(mock_vector_store, mock_memory_manager):
    """RAGChain with all external dependencies mocked."""
    with patch("app.services.rag_chain.ChatOpenAI"), \
         patch("app.services.rag_chain.get_settings") as mock_settings:

        settings = MagicMock()
        settings.llm_model = "gpt-3.5-turbo"
        settings.openai_api_key = "test-key"
        settings.max_tokens = 500
        settings.debug = False
        settings.top_k_results = 2
        mock_settings.return_value = settings

        from app.services.rag_chain import RAGChain
        chain = RAGChain(
            vector_store=mock_vector_store,
            memory_manager=mock_memory_manager,
        )
        return chain


# ─────────────────────────────────────────────────────
# Tests — Empty Store Guard
# ─────────────────────────────────────────────────────

class TestEmptyStoreGuard:
    def test_raises_runtime_error_when_store_empty(
        self, mock_vector_store, mock_memory_manager
    ):
        """RuntimeError raised when no documents have been ingested."""
        mock_vector_store.collection_is_empty.return_value = True

        with patch("app.services.rag_chain.ChatOpenAI"), \
             patch("app.services.rag_chain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_model = "gpt-3.5-turbo"
            settings.openai_api_key = "test-key"
            settings.max_tokens = 500
            settings.debug = False
            mock_settings.return_value = settings

            from app.services.rag_chain import RAGChain
            chain = RAGChain(
                vector_store=mock_vector_store,
                memory_manager=mock_memory_manager,
            )

            with pytest.raises(RuntimeError, match="No documents found"):
                chain.query("How many leave days?", session_id="session_1")


# ─────────────────────────────────────────────────────
# Tests — Query Response Structure
# ─────────────────────────────────────────────────────

class TestQueryResponse:
    def test_query_returns_dict_with_required_keys(self, rag_chain):
        """query() returns dict with answer, sources, session_id."""
        with patch.object(rag_chain, "_build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "You get 20 days annual leave.",
                "source_documents": make_mock_source_docs(),
            }
            mock_build.return_value = mock_chain

            result = rag_chain.query("How many leave days?", "session_abc")

        assert "answer" in result
        assert "sources" in result
        assert "session_id" in result

    def test_session_id_echoed_in_response(self, rag_chain):
        """session_id in response matches the one passed in."""
        with patch.object(rag_chain, "_build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "Some answer.",
                "source_documents": [],
            }
            mock_build.return_value = mock_chain

            result = rag_chain.query("Any question", "my_session_99")

        assert result["session_id"] == "my_session_99"

    def test_sources_contain_source_and_page(self, rag_chain):
        """Each source citation has 'source' and 'page' keys."""
        with patch.object(rag_chain, "_build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "Answer here.",
                "source_documents": make_mock_source_docs(),
            }
            mock_build.return_value = mock_chain

            result = rag_chain.query("Leave policy?", "s1")

        for source in result["sources"]:
            assert "source" in source
            assert "page" in source

    def test_duplicate_sources_deduplicated(self, rag_chain):
        """Same source/page combination appears only once in citations."""
        duplicate_docs = [
            Document(
                page_content="Content A",
                metadata={"source": "policy.pdf", "page": 1},
            ),
            Document(
                page_content="Content B",
                metadata={"source": "policy.pdf", "page": 1},
            ),
        ]
        with patch.object(rag_chain, "_build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "Answer.",
                "source_documents": duplicate_docs,
            }
            mock_build.return_value = mock_chain

            result = rag_chain.query("Anything?", "s2")

        # Should have only 1 unique citation, not 2
        assert len(result["sources"]) == 1

    def test_no_source_documents_returns_empty_sources(self, rag_chain):
        """Empty source_documents returns empty sources list."""
        with patch.object(rag_chain, "_build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "Not found in documents.",
                "source_documents": [],
            }
            mock_build.return_value = mock_chain

            result = rag_chain.query("Obscure question", "s3")

        assert result["sources"] == []


# ─────────────────────────────────────────────────────
# Tests — Source Extraction
# ─────────────────────────────────────────────────────

class TestSourceExtraction:
    def test_extract_sources_returns_list(self, rag_chain):
        """_extract_sources returns a list."""
        sources = rag_chain._extract_sources(make_mock_source_docs())
        assert isinstance(sources, list)

    def test_extract_sources_empty_input(self, rag_chain):
        """_extract_sources handles empty input gracefully."""
        sources = rag_chain._extract_sources([])
        assert sources == []
