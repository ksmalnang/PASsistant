"""Common helpers for API routes."""

from fastapi import UploadFile

from src.utils.nodes import DocumentProcessingNode


async def read_upload_files(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read uploaded files into memory as filename/content tuples."""
    file_tuples: list[tuple[str, bytes]] = []
    for file in files:
        contents = await file.read()
        file_tuples.append((file.filename, contents))
    return file_tuples


def create_document_processor():
    """Create the default document processor."""
    return DocumentProcessingNode()
