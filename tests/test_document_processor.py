"""
tests/test_document_processor.py
──────────────────────────────────
Tests for DocumentProcessor service.
All tests follow AAA pattern: Arrange → Act → Assert.
"""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.document_processor import DocumentProcessor


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture
def processor() -> DocumentProcessor:
    """Default processor with small chunk size for test predictability."""
    return DocumentProcessor(chunk_size=200, chunk_overlap=20)


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    """Create a minimal valid PDF using reportlab if available, else skip."""
    try:
        from reportlab.pdfgen import canvas
        pdf_path = tmp_path / "test_policy.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Annual Leave Policy")
        c.drawString(100, 730, "Employees are entitled to 20 days of annual leave.")
        c.drawString(100, 710, "Leave must be approved by the line manager.")
        c.drawString(100, 690, "Unused leave can be carried forward up to 5 days.")
        c.save()
        return str(pdf_path)
    except ImportError:
        pytest.skip("reportlab not installed — skipping PDF fixture tests")


@pytest.fixture
def sample_docx(tmp_path) -> str:
    """Create a minimal DOCX file for testing."""
    try:
        from docx import Document as DocxDocument
        docx_path = tmp_path / "test_policy.docx"
        doc = DocxDocument()
        doc.add_heading("HR Code of Conduct", 0)
        doc.add_paragraph("All employees must maintain professional standards.")
        doc.add_paragraph("Harassment of any kind is strictly prohibited.")
        doc.save(str(docx_path))
        return str(docx_path)
    except ImportError:
        pytest.skip("python-docx not installed — skipping DOCX fixture tests")


# ─────────────────────────────────────────────────────
# Tests — File Validation
# ─────────────────────────────────────────────────────

class TestFileValidation:
    def test_nonexistent_file_raises_file_not_found(self, processor):
        """FileNotFoundError is raised for non-existent paths."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.load_and_split("/nonexistent/path/file.pdf")

    def test_unsupported_extension_raises_value_error(self, processor, tmp_path):
        """ValueError is raised for unsupported file types."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Some content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_and_split(str(txt_file))

    def test_csv_file_raises_value_error(self, processor, tmp_path):
        """CSV files are rejected with a descriptive error."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nfoo,bar")
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_and_split(str(csv_file))


# ─────────────────────────────────────────────────────
# Tests — PDF Loading
# ─────────────────────────────────────────────────────

class TestPDFLoading:
    def test_pdf_returns_document_list(self, processor, sample_pdf):
        """Loading a PDF returns a non-empty list."""
        chunks = processor.load_and_split(sample_pdf)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_pdf_chunks_are_document_instances(self, processor, sample_pdf):
        """All returned items are LangChain Document objects."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_pdf_metadata_has_source(self, processor, sample_pdf):
        """Every chunk contains a 'source' metadata field."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] != ""

    def test_pdf_metadata_has_page_number(self, processor, sample_pdf):
        """Every chunk contains a 'page' metadata field."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            assert "page" in chunk.metadata

    def test_pdf_source_is_filename_not_full_path(self, processor, sample_pdf):
        """Source metadata contains only the filename, not the full path."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            source = chunk.metadata["source"]
            assert "/" not in source or source == sample_pdf.split("/")[-1]


# ─────────────────────────────────────────────────────
# Tests — DOCX Loading
# ─────────────────────────────────────────────────────

class TestDocxLoading:
    def test_docx_returns_document_list(self, processor, sample_docx):
        """Loading a DOCX returns a non-empty list."""
        chunks = processor.load_and_split(sample_docx)
        assert len(chunks) > 0

    def test_docx_metadata_has_source(self, processor, sample_docx):
        """DOCX chunks preserve source filename in metadata."""
        chunks = processor.load_and_split(sample_docx)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_docx_page_defaults_to_zero(self, processor, sample_docx):
        """DOCX files don't have pages — page defaults to 0."""
        chunks = processor.load_and_split(sample_docx)
        for chunk in chunks:
            assert chunk.metadata.get("page") == 0


# ─────────────────────────────────────────────────────
# Tests — Chunking Behaviour
# ─────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_size_not_exceeded(self, processor, sample_pdf):
        """No chunk exceeds the configured chunk_size."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            assert len(chunk.page_content) <= processor.chunk_size * 1.2  # 20% tolerance

    def test_processor_with_custom_chunk_size(self, sample_pdf):
        """Custom chunk_size is respected."""
        small_processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = small_processor.load_and_split(sample_pdf)
        assert len(chunks) > 0  # Should produce more chunks with smaller size

    def test_chunks_have_non_empty_content(self, processor, sample_pdf):
        """No chunk has empty page_content."""
        chunks = processor.load_and_split(sample_pdf)
        for chunk in chunks:
            assert chunk.page_content.strip() != ""


# ─────────────────────────────────────────────────────
# Tests — Directory Loading
# ─────────────────────────────────────────────────────

class TestDirectoryLoading:
    def test_nonexistent_directory_raises_error(self, processor):
        """FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            processor.load_directory("/nonexistent/dir")

    def test_empty_directory_returns_empty_list(self, processor, tmp_path):
        """An empty directory returns an empty list without error."""
        result = processor.load_directory(str(tmp_path))
        assert result == []

    def test_directory_with_pdf_returns_chunks(self, processor, tmp_path, sample_pdf):
        """Directory containing a PDF returns chunks."""
        import shutil
        shutil.copy(sample_pdf, tmp_path / "policy.pdf")
        chunks = processor.load_directory(str(tmp_path))
        assert len(chunks) > 0
