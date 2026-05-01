"""Local raw-upload storage helpers."""

import logging
import mimetypes
import re
import uuid
from pathlib import Path

from src.config import get_settings
from src.utils.state import DocumentType, DocumentUpload

logger = logging.getLogger(__name__)


class DocumentTools:
    """Persist raw uploaded files and infer lightweight metadata.

    This helper only writes uploads to local storage and classifies filenames.
    It does not run OCR, chunk documents, or write anything to the vector store.
    """

    _DOCUMENT_TYPE_KEYWORDS: tuple[tuple[DocumentType, tuple[str, ...]], ...] = (
        (
            DocumentType.TRANSCRIPT,
            ("transcript", "grades", "grade", "academic record", "nilai", "khs"),
        ),
        (
            DocumentType.ID_CARD,
            ("id card", "identity", "ktp", "passport", "license", "student card", "kartu mahasiswa"),
        ),
        (
            DocumentType.APPLICATION,
            ("application", "apply", "admission", "registration form", "formulir pendaftaran", "form"),
        ),
        (
            DocumentType.RECOMMENDATION,
            ("recommendation", "reference letter", "referral letter"),
        ),
        (
            DocumentType.CURRICULUM,
            ("curriculum", "kurikulum", "cpl", "capaian pembelajaran", "profil lulusan"),
        ),
        (
            DocumentType.SYLLABUS,
            ("syllabus", "silabus", "rps", "course outline", "semester plan"),
        ),
        (
            DocumentType.CERTIFICATE,
            ("certificate", "sertifikat", "ijazah", "diploma", "surat keterangan"),
        ),
        (
            DocumentType.INVOICE,
            ("invoice", "billing", "receipt", "tuition bill", "payment slip", "kwitansi"),
        ),
        (
            DocumentType.POLICY,
            ("policy", "regulation", "guideline", "handbook", "peraturan", "kebijakan", "panduan"),
        ),
    )

    def __init__(self):
        settings = get_settings()
        self.raw_dir = settings.raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
    ) -> DocumentUpload:
        """
        Save an uploaded file to raw storage only.

        Args:
            file_bytes: Raw file bytes
            original_filename: Original filename

        Returns:
            DocumentUpload with local storage metadata
        """
        document_id = str(uuid.uuid4())
        file_extension = Path(original_filename).suffix
        storage_filename = f"{document_id}{file_extension}"
        storage_path = self.raw_dir / storage_filename
        storage_path.write_bytes(file_bytes)

        mime_type, _ = mimetypes.guess_type(original_filename)
        document = DocumentUpload(
            document_id=document_id,
            filename=original_filename,
            file_path=str(storage_path),
            document_type=self._detect_document_type(original_filename),
            mime_type=mime_type or "application/octet-stream",
            file_size=len(file_bytes),
        )

        logger.info(
            "Document saved",
            extra={
                "document_id": document_id,
                "uploaded_filename": original_filename,
                "file_size_bytes": len(file_bytes),
            },
        )
        return document

    def _detect_document_type(self, filename: str) -> DocumentType:
        """Infer document type from filename keywords."""
        normalized_name = " ".join(re.findall(r"[a-z0-9]+", filename.lower()))
        tokens = set(normalized_name.split())

        for document_type, keywords in self._DOCUMENT_TYPE_KEYWORDS:
            if self._matches_keywords(normalized_name, tokens, keywords):
                return document_type

        return DocumentType.OTHER

    def _matches_keywords(
        self,
        normalized_name: str,
        tokens: set[str],
        keywords: tuple[str, ...],
    ) -> bool:
        """Return whether a normalized filename matches any keyword or phrase."""
        for keyword in keywords:
            normalized_keyword = " ".join(re.findall(r"[a-z0-9]+", keyword.lower()))
            if not normalized_keyword:
                continue
            if " " in normalized_keyword:
                if normalized_keyword in normalized_name:
                    return True
                continue
            if normalized_keyword in tokens:
                return True
        return False

    def delete_file(self, file_path: str) -> None:
        """Delete a stored file."""
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info("Deleted file: %s", file_path)
