"""
Production readiness tests for the ingestion and retrieval pipeline.

These tests validate that the chunking, indexing, and retrieval pipeline
correctly handles the academic document patterns found in the knowledge base
(Buku Panduan, Kurikulum IF, Buku Pedoman Kemahasiswaan).

Run with: uv run python -m pytest tests/test_prod_readiness.py -v
"""

import re

import pytest

from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import DocumentType, DocumentUpload
from src.utils.tools.hierarchical_chunking import (
    SECTION_TYPES,
    HierarchicalChunker,
)
from src.utils.vector_store.indexing import IndexingOperations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KURIKULUM_SEMESTER_TABLE = """
## 6 Struktur Matakuliah dlm Kurikulum Program Studi

## 6.1 Matrik Kurikulum

<table border="1"><tr><td>Smt</td><td>sks</td></tr><tr><td>VIII</td><td>12</td></tr></table>

## 7 Daftar sebaran mata kuliah tiap semester
## Tabel 7. Daftar Mata kuliah per semester
<table border="1"><tr><td colspan="7">Semester 01</td></tr><tr><td>1</td><td>IF21W0101</td><td>Agama</td><td>2</td><td></td><td></td><td>2</td></tr></table>

## 8 Implementasi Hak Belajar

## 8.1 Model implementasi MBKM

Content about MBKM.
""".strip()

BUKU_PANDUAN_PRODI_IF = """
## III.4. Program Studi Teknik Informatika

Program Studi Teknik Informatika

Semester I

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100101</td><td>Agama</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>19</td></tr></table>

Semester II

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100201</td><td>Pendidikan Pancasila</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>19</td></tr></table>

Semester V

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100501</td><td>Bahasa Indonesia</td><td>MKU</td><td>2</td></tr><tr><td>2</td><td>IF2100502</td><td>Internet of Things</td><td>MKK</td><td>2</td></tr><tr><td>3</td><td>IF2100503</td><td>Intelegensia Buatan</td><td>MKK</td><td>3</td></tr><tr><td colspan="4">Jumlah SKS</td><td>20</td></tr></table>

Semester VIII

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100801</td><td>Islam Dasar Ilmu</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>12</td></tr></table>

Pilihan 1

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>SKS</td></tr><tr><td>1</td><td>IF2110501</td><td>Algoritma Optimasi</td><td>3</td></tr><tr><td>2</td><td>IF2110502</td><td>Data Warehouse</td><td>3</td></tr></table>
""".strip()

BUKU_PANDUAN_DPP = """
## IV.1. Registrasi dan Perwalian

## IV.1.1. Tahap Pendaftaran dan Pembayaran DPP

Seluruh mahasiswa diwajibkan melakukan perwalian, termasuk yang sedang melakukan kerja praktek. Perwalian dapat dilakukan setelah mahasiswa memenuhi persyaratan administrasi pembayaran uang kuliah yang disyaratkan dari DPP/SPP Tahun Akademik yang bersangkutan.

DPP dana pelaksanaan pendidikan : dibayarkan 4 kali cicilan
- Cicilan I 25% : Juli
- Cicilan II 25% : Oktober
- Cicilan III 25% : Januari
- Cicilan IV 25%: April

Cara pembayaran DPP di Unpas dilayani melalui
a. Kasir Unpas di kampus Taman Sari
b. Online melalui Sevimapay, yang di dalamnya terdapat kanal
i. Tokopedia
ii. Shopee
iii. Mandiri

## IV.1.2. Tahap Perwalian Reguler

Perwalian dilakukan secara online melalui SITU 2.0.
""".strip()

BUKU_PANDUAN_KP = """
## IV.4. Mata Kuliah Kerja Praktek (KP)

Kerja praktek bertujuan untuk mengenalkan mahasiswa kepada lingkungan praktis.

Prosedur seminar kerja praktek:
- Tercantum dalam kartu rencana studi
- Mahasiswa mendaftarkan ke coordinator kerja praktek

Jika mahasiswa peserta kerja praktek sudah melakukan bimbingan dengan pembimbing yang ditunjuk koordinator kerja praktek tapi mahasiswa tidak aktif (tidak bimbingan dan tidak registrasi ulang ke koordinator) minimal 2 semester, maka topik kerja praktek yang bersangkutan dinyatakan kadaluarsa dan harus diganti topik baru.

## IV.5. Tugas Akhir

Tugas akhir merupakan syarat kelulusan.
""".strip()


def _build_doc(text: str, filename: str = "test.pdf") -> DocumentUpload:
    return DocumentUpload(
        document_id="test-doc-001",
        filename=filename,
        file_path="C:/tmp/test.pdf",
        document_type=DocumentType.OTHER,
        mime_type="application/pdf",
        file_size=1024,
        extracted_text=text,
    )


# ---------------------------------------------------------------------------
# 1. Heading Detection Tests
# ---------------------------------------------------------------------------


class TestHeadingDetection:
    """Verify heading patterns detect academic document structures."""

    def setup_method(self):
        self.chunker = HierarchicalChunker()

    def test_semester_roman_detected_as_subsection(self):
        """'Semester V' standalone should be detected as a subsection."""
        result = self.chunker._detect_heading("Semester V")
        assert result is not None
        assert result[0] == "subsection"
        assert result[1] == "v"

    def test_semester_viii_detected(self):
        result = self.chunker._detect_heading("Semester VIII")
        assert result is not None
        assert result[0] == "subsection"
        assert result[1] == "viii"

    def test_semester_with_markdown_prefix(self):
        result = self.chunker._detect_heading("## Semester III")
        assert result is not None
        assert result[0] == "subsection"

    def test_pilihan_detected_as_clause(self):
        result = self.chunker._detect_heading("Pilihan 1")
        assert result is not None
        assert result[0] == "clause"
        assert result[1] == "1"

    def test_pilihan_3_detected(self):
        result = self.chunker._detect_heading("Pilihan 3")
        assert result is not None
        assert result[0] == "clause"

    def test_mid_sentence_semester_not_detected(self):
        """'Semester' inside a sentence should NOT be detected as heading."""
        result = self.chunker._detect_heading(
            "Pada Semester V mahasiswa mengambil mata kuliah pilihan"
        )
        assert result is None

    def test_numeric_chapter_detected(self):
        result = self.chunker._detect_heading("7 Daftar sebaran mata kuliah tiap semester")
        assert result is not None
        assert result[0] == "chapter"
        assert result[1] == "7"

    def test_roman_section_detected(self):
        result = self.chunker._detect_heading("IV.1.1. Tahap Pendaftaran dan Pembayaran DPP")
        assert result is not None


# ---------------------------------------------------------------------------
# 2. Chunking Structure Tests
# ---------------------------------------------------------------------------


class TestChunkingStructure:
    """Verify documents are chunked into correct hierarchical structures."""

    def setup_method(self):
        self.chunker = HierarchicalChunker()

    def test_semester_headings_create_separate_subsections(self):
        """Each 'Semester X' should become its own subsection under the parent."""
        doc = _build_doc(BUKU_PANDUAN_PRODI_IF)
        structured = self.chunker.chunk_document(doc)

        parent_sections = {p.section_id for p in structured.parent_chunks}
        # Should have subsections for each semester
        semester_sections = [s for s in parent_sections if "subbagian" in s]
        assert len(semester_sections) >= 4, (
            f"Expected at least 4 semester subsections, got {len(semester_sections)}: {semester_sections}"
        )

    def test_semester_v_chunk_contains_correct_courses(self):
        """Semester V subsection should contain IF2100501, IF2100502, IF2100503."""
        doc = _build_doc(BUKU_PANDUAN_PRODI_IF)
        structured = self.chunker.chunk_document(doc)

        semester_v_chunks = [
            c for c in structured.child_chunks
            if "subbagian_v" in c.metadata.get("section_id", "")
        ]
        assert semester_v_chunks, "No child chunks found for Semester V"
        combined_text = " ".join(c.text for c in semester_v_chunks)
        assert "IF2100501" in combined_text
        assert "Bahasa Indonesia" in combined_text
        assert "Internet of Things" in combined_text or "Internet of Think" in combined_text

    def test_pilihan_creates_separate_chunk(self):
        """'Pilihan 1' should become its own clause-level chunk."""
        doc = _build_doc(BUKU_PANDUAN_PRODI_IF)
        structured = self.chunker.chunk_document(doc)

        pilihan_chunks = [
            c for c in structured.child_chunks
            if "pilihan" in c.metadata.get("section_id", "").lower()
            or "Algoritma Optimasi" in c.text
        ]
        assert pilihan_chunks, "No child chunks found for Pilihan 1"
        combined_text = " ".join(c.text for c in pilihan_chunks)
        assert "Algoritma Optimasi" in combined_text
        assert "Data Warehouse" in combined_text

    def test_leaf_chapter_is_indexed(self):
        """Chapter 7 (no sub-sections) should still be included as a parent chunk."""
        doc = _build_doc(KURIKULUM_SEMESTER_TABLE)
        structured = self.chunker.chunk_document(doc)

        parent_sections = {p.section_id for p in structured.parent_chunks}
        assert "bab_7" in parent_sections, (
            f"Leaf chapter 'bab_7' not in parent chunks: {parent_sections}"
        )

    def test_leaf_chapter_content_is_indexed_as_children(self):
        """Leaf chapter 7 content (table) should produce child chunks."""
        doc = _build_doc(KURIKULUM_SEMESTER_TABLE)
        structured = self.chunker.chunk_document(doc)

        bab7_children = [
            c for c in structured.child_chunks
            if "bab_7" in c.parent_id
        ]
        assert bab7_children, "No child chunks produced for leaf chapter bab_7"
        combined_text = " ".join(c.text for c in bab7_children)
        assert "IF21W0101" in combined_text or "Agama" in combined_text

    def test_dpp_payment_info_fully_indexed(self):
        """DPP payment section should contain Tokopedia, Sevimapay, cicilan info."""
        doc = _build_doc(BUKU_PANDUAN_DPP)
        structured = self.chunker.chunk_document(doc)

        all_child_text = " ".join(c.text for c in structured.child_chunks)
        assert "Sevimapay" in all_child_text
        assert "Tokopedia" in all_child_text
        assert "cicilan" in all_child_text.lower() or "Cicilan" in all_child_text

    def test_kp_kadaluarsa_rule_indexed(self):
        """Kerja Praktek section should contain the 'kadaluarsa' rule."""
        doc = _build_doc(BUKU_PANDUAN_KP)
        structured = self.chunker.chunk_document(doc)

        all_child_text = " ".join(c.text for c in structured.child_chunks)
        assert "kadaluarsa" in all_child_text

    def test_parent_max_chars_increased(self):
        """Default parent_max_chars should be 8000 (not 5000)."""
        chunker = HierarchicalChunker()
        assert chunker.parent_max_chars == 8000


# ---------------------------------------------------------------------------
# 3. Contextual Embedding Tests
# ---------------------------------------------------------------------------


class TestContextualEmbedding:
    """Verify that breadcrumb context is prepended to embedding text."""

    def test_build_embedding_text_prepends_breadcrumb(self):
        """Child chunk embedding text should include breadcrumb for context."""
        doc = _build_doc(BUKU_PANDUAN_PRODI_IF)
        chunker = HierarchicalChunker()
        structured = chunker.chunk_document(doc)

        # Find a Semester V child chunk
        semester_v_chunks = [
            c for c in structured.child_chunks
            if "subbagian_v" in c.metadata.get("section_id", "")
        ]
        assert semester_v_chunks

        chunk = semester_v_chunks[0]
        indexer = IndexingOperations.__new__(IndexingOperations)
        embedding_text = indexer._build_embedding_text(chunk)

        # Embedding text should contain breadcrumb (which includes prodi name)
        assert "Teknik Informatika" in embedding_text
        assert "Semester V" in embedding_text
        # And also the actual chunk content
        assert chunk.text.strip()[:50] in embedding_text

    def test_embedding_text_without_breadcrumb_returns_raw_text(self):
        """Chunks without breadcrumb should return raw text."""
        from src.utils.tools.hierarchical_chunking import ChildChunk

        chunk = ChildChunk(
            chunk_id="test",
            parent_id="parent",
            text="Some raw content",
            metadata={},
        )
        indexer = IndexingOperations.__new__(IndexingOperations)
        embedding_text = indexer._build_embedding_text(chunk)
        assert embedding_text == "Some raw content"


# ---------------------------------------------------------------------------
# 4. Query Normalization Tests
# ---------------------------------------------------------------------------


class TestQueryNormalization:
    """Verify semester numeral normalization and query expansion."""

    def setup_method(self):
        self.node = RetrievalNode.__new__(RetrievalNode)

    def test_arabic_to_roman_semester_5(self):
        result = self.node._normalize_semester_numerals("matakuliah semester 5 teknik informatika")
        assert "semester V" in result

    def test_arabic_to_roman_semester_8(self):
        result = self.node._normalize_semester_numerals("jadwal semester 8")
        assert "semester VIII" in result

    def test_roman_to_arabic_semester_v(self):
        result = self.node._normalize_semester_numerals("matakuliah semester V teknik informatika")
        assert "semester 5" in result

    def test_roman_to_arabic_semester_iii(self):
        result = self.node._normalize_semester_numerals("kuliah semester III")
        assert "semester 3" in result

    def test_no_semester_unchanged(self):
        result = self.node._normalize_semester_numerals("syarat perwalian DPP")
        assert result == "syarat perwalian DPP"

    def test_expand_query_normalizes_semester(self):
        """_expand_query should produce a semester-normalized variant."""
        result = self.node._expand_query("matakuliah semester 5 teknik informatika")
        assert "V" in result or "semester V" in result.lower()

    def test_expand_query_does_not_add_semester_berturut(self):
        """Removed expansion should not appear."""
        result = self.node._expand_query("jadwal semester 5")
        assert "berturut" not in result


# ---------------------------------------------------------------------------
# 5. Retrieval Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestRetrievalPipeline:
    """Verify retrieval node behavior and confidence scoring."""

    def test_policy_scope_terms_are_specific(self):
        """Generic terms like 'bisa', 'boleh' should NOT be in policy scope."""
        node = RetrievalNode.__new__(RetrievalNode)
        assert "bisa" not in node._POLICY_SCOPE_TERMS
        assert "boleh" not in node._POLICY_SCOPE_TERMS
        assert "kapan" not in node._POLICY_SCOPE_TERMS
        assert "bulan" not in node._POLICY_SCOPE_TERMS

    def test_policy_scope_terms_include_financial(self):
        """Financial/policy terms should be in scope."""
        node = RetrievalNode.__new__(RetrievalNode)
        assert "dpp" in node._POLICY_SCOPE_TERMS
        assert "cicilan" in node._POLICY_SCOPE_TERMS
        assert "perwalian" in node._POLICY_SCOPE_TERMS
        assert "pembayaran" in node._POLICY_SCOPE_TERMS

    def test_looks_like_policy_scope_detects_dpp(self):
        node = RetrievalNode.__new__(RetrievalNode)
        assert node._looks_like_policy_scope("telat bayar DPP cicilan pertama")

    def test_looks_like_policy_scope_rejects_generic(self):
        node = RetrievalNode.__new__(RetrievalNode)
        assert not node._looks_like_policy_scope("apa saja matakuliah semester 5")

    def test_extract_keywords_removes_stopwords(self):
        node = RetrievalNode.__new__(RetrievalNode)
        result = node._extract_keywords("Kak, kalau aku telat bayar DPP bisa ikut perwalian nggak?")
        assert "kak" not in result
        assert "aku" not in result
        assert "nggak" not in result
        assert "telat" in result
        assert "perwalian" in result

    def test_score_threshold_is_0_2(self):
        """Score threshold should be 0.2 (not the old 0.4)."""
        # Verify the hardcoded threshold in the run method
        import inspect
        source = inspect.getsource(RetrievalNode.run)
        assert "score_threshold=0.2" in source


# ---------------------------------------------------------------------------
# 6. BM25 Configuration Tests
# ---------------------------------------------------------------------------


class TestBM25Configuration:
    """Verify BM25 stemmer is enabled for Indonesian text."""

    def test_bm25_stemmer_enabled(self):
        """BM25_DISABLE_STEMMER should be False for Indonesian morphology support."""
        from src.utils.vector_store.bm25 import BM25_DISABLE_STEMMER
        assert BM25_DISABLE_STEMMER is False


# ---------------------------------------------------------------------------
# 7. Document Management API Tests
# ---------------------------------------------------------------------------


class TestDocumentManagementAPI:
    """Verify document list and delete endpoints exist and are configured."""

    def test_list_documents_endpoint_exists(self):
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/documents")
        # Should not 404 — may return empty list or actual docs
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_endpoint_returns_404_for_missing_file(self):
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.delete("/documents/by-filename/nonexistent_file.pdf")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 8. Ingestion Health Check Tests
# ---------------------------------------------------------------------------


class TestIngestionHealthCheck:
    """Verify ingestion health checks catch common issues."""

    def test_low_chunk_density_detected(self):
        """Documents with too few parent chunks per page should raise ERROR."""
        from src.services.ingestion_health import IngestionHealthCheck

        health = IngestionHealthCheck()
        doc = _build_doc("Short text.")
        doc.num_pages = 50  # 50 pages but very little content

        chunker = HierarchicalChunker()
        structured = chunker.chunk_document(doc)
        report = health.validate(doc, structured)

        error_codes = [issue.code for issue in report.issues]
        assert "LOW_CHUNK_DENSITY" in error_codes or "TOO_FEW_CHILD_CHUNKS" in error_codes
