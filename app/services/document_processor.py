"""
app/services/document_processor.py
────────────────────────────────────
Handles loading and chunking of PDF and DOCX files.
Preserves metadata (source filename, page number) on every chunk.

Class:
    DocumentProcessor — load, validate, and split documents into chunks.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class DocumentProcessor:
    """
    Loads PDF/DOCX files and splits them into overlapping chunks
    suitable for embedding and vector storage.

    Attributes:
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap characters between consecutive chunks.
        splitter (RecursiveCharacterTextSplitter): Text splitting engine.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"DocumentProcessor initialised | chunk_size={self.chunk_size} "
            f"chunk_overlap={self.chunk_overlap}"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def load_and_split(self, file_path: str) -> List[Document]:
        """
        Load a single PDF or DOCX file and return a list of chunks.

        Each chunk's metadata contains:
            - source: original filename
            - page: page number (PDF only, else 0)
            - doc_type: "pdf" or "docx"

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            List of LangChain Document objects (chunks).

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file extension is not supported.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info(f"Loading document: {path.name}")
        raw_docs = self._load_file(path)
        chunks = self._split_documents(raw_docs, source_name=path.name)

        logger.info(f"Produced {len(chunks)} chunks from '{path.name}'")
        return chunks

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        Load all supported files from a directory.

        Args:
            dir_path: Path to the directory.

        Returns:
            Combined list of chunks from all supported files.

        Raises:
            FileNotFoundError: If dir_path does not exist.
        """
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_chunks: List[Document] = []
        found_files = [
            f for f in directory.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not found_files:
            logger.warning(f"No supported files found in {dir_path}")
            return all_chunks

        for file_path in found_files:
            try:
                chunks = self.load_and_split(str(file_path))
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(f"Failed to process {file_path.name}: {exc}")

        logger.info(
            f"Directory ingestion complete | "
            f"files={len(found_files)} total_chunks={len(all_chunks)}"
        )
        return all_chunks

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    def _validate_file(self, path: Path) -> None:
        """Raise descriptive errors for invalid file paths."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

    def _load_file(self, path: Path) -> List[Document]:
        """Dispatch to the correct loader based on file extension."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            # Ensure page metadata exists
            for i, doc in enumerate(docs):
                doc.metadata.setdefault("page", i)
                doc.metadata["doc_type"] = "pdf"
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["page"] = 0
                doc.metadata["doc_type"] = "docx"
        else:
            raise ValueError(f"No loader for extension: {suffix}")
        return docs

    def _split_documents(
        self,
        documents: List[Document],
        source_name: str,
    ) -> List[Document]:
        """
        Split raw documents into chunks, injecting source metadata.
        """
        chunks = self.splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = source_name
        return chunks
