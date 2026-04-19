"""
app/services/vector_store.py
─────────────────────────────
Manages the ChromaDB vector store:
  - Embedding model initialisation (OpenAI or HuggingFace)
  - Storing document chunks
  - Similarity search retrieval

Class:
    VectorStoreManager — wraps ChromaDB with a clean interface.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _build_embedding_function():
    """
    Factory: return the correct embedding model based on config.

    Returns OpenAI embeddings by default; falls back to HuggingFace
    sentence-transformers when EMBEDDING_PROVIDER=huggingface.
    """
    provider = settings.embedding_provider.lower()

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info(
            f"Using HuggingFace embeddings: {settings.huggingface_model}"
        )
        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Default: OpenAI
    from langchain_openai import OpenAIEmbeddings
    logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


class VectorStoreManager:
    """
    Wrapper around ChromaDB for storing and retrieving
    embedded document chunks.

    Attributes:
        collection_name (str): Name of the ChromaDB collection.
        persist_dir (str): Directory for persistent storage.
        embeddings: LangChain-compatible embedding function.
        _store (Chroma | None): Lazy-loaded Chroma instance.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.embeddings = _build_embedding_function()
        self._store: Optional[Chroma] = None

        logger.info(
            f"VectorStoreManager initialised | "
            f"collection='{self.collection_name}' "
            f"persist_dir='{self.persist_dir}'"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and store a list of document chunks.

        Args:
            documents: List of LangChain Document chunks.

        Returns:
            Number of documents successfully stored.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot add an empty document list.")

        store = self._get_or_create_store()
        store.add_documents(documents)

        logger.info(f"Added {len(documents)} chunks to ChromaDB.")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve the top-k most semantically similar chunks.

        Args:
            query: The user's natural language question.
            k: Number of results to return (default from settings).

        Returns:
            List of Document chunks ranked by similarity.

        Raises:
            RuntimeError: If the store is empty.
        """
        top_k = k or settings.top_k_results
        store = self._get_or_create_store()

        results = store.similarity_search(query, k=top_k)
        logger.info(
            f"Similarity search | query='{query[:60]}...' "
            f"returned={len(results)} chunks"
        )
        return results

    def get_retriever(self, k: Optional[int] = None):
        """
        Return a LangChain-compatible retriever for use in chains.

        Args:
            k: Number of documents to retrieve per query.

        Returns:
            VectorStoreRetriever instance.
        """
        top_k = k or settings.top_k_results
        store = self._get_or_create_store()
        return store.as_retriever(search_kwargs={"k": top_k})

    def collection_is_empty(self) -> bool:
        """Return True if no documents have been ingested yet."""
        try:
            store = self._get_or_create_store()
            count = store._collection.count()
            return count == 0
        except Exception:
            return True

    def reset_collection(self) -> None:
        """
        Delete all documents from the collection.
        Useful for testing and re-ingestion.
        """
        store = self._get_or_create_store()
        store._collection.delete(where={"source": {"$ne": ""}})
        self._store = None
        logger.warning(f"Collection '{self.collection_name}' has been reset.")

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    def _get_or_create_store(self) -> Chroma:
        """
        Lazy-load the Chroma store.
        Creates it on first call; returns cached instance thereafter.
        """
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._store
