"""Document upload endpoints."""

import logging
import mimetypes
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.models import (
    DocumentIngestionResponse,
    ErrorResponse,
)
from src.api.services import handle_knowledge_base_ingestion
from src.utils.tools.ocr import PDF_MIME_TYPE, SUPPORTED_MIME_TYPES, GLMOCRTool

logger = logging.getLogger(__name__)
router = APIRouter()
UploadFiles = Annotated[
    list[UploadFile],
    File(
        ...,
        description="One or more PDF or image files to ingest into the retrieval index.",
    ),
]
PDF_MIME_ALIASES = {"application/x-pdf", "application/acrobat"}
DOCUMENT_ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The upload request is missing a required file payload.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "One or more uploaded files exceed the supported size limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": "One or more uploaded files use an unsupported MIME type.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The server failed to process the uploaded documents.",
    },
}


def _normalize_mime_type(content_type: str | None, filename: str | None) -> str | None:
    """Normalize declared or inferred MIME types into the OCR-supported set."""
    candidates = [content_type]
    guessed_type, _ = mimetypes.guess_type(filename or "")
    candidates.append(guessed_type)

    for candidate in candidates:
        if not candidate:
            continue
        if candidate in PDF_MIME_ALIASES:
            return PDF_MIME_TYPE
        if candidate in SUPPORTED_MIME_TYPES:
            return candidate
    return None


def _resolve_upload_size(file: UploadFile) -> int:
    """Resolve upload size without consuming the request body stream."""
    if getattr(file, "size", None) is not None:
        return int(file.size)

    stream = file.file
    current_position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(current_position)
    return int(size)


def _validate_files(files: list[UploadFile]) -> list[UploadFile]:
    """Validate multipart uploads before downstream processing."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be uploaded.",
        )

    for file in files:
        filename = file.filename or "upload"
        mime_type = _normalize_mime_type(file.content_type, filename)
        if mime_type is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file type for '{filename}'. "
                    "Only PDF, PNG, JPEG, GIF, and WEBP uploads are supported."
                ),
            )

        size_bytes = _resolve_upload_size(file)
        max_bytes = (
            GLMOCRTool.MAX_PDF_BYTES if mime_type == PDF_MIME_TYPE else GLMOCRTool.MAX_IMAGE_BYTES
        )
        if size_bytes > max_bytes:
            max_size_mb = max_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Uploaded file '{filename}' is too large. "
                    f"Maximum size for {mime_type} is {max_size_mb:.1f} MB."
                ),
            )

    return files


async def validate_document_upload_files(files: UploadFiles) -> list[UploadFile]:
    """Validate multipart file uploads for ingestion endpoints."""
    return _validate_files(files)


@router.post(
    "/upload",
    response_model=list[DocumentIngestionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest documents",
    description=(
        "Upload one or more documents, run OCR and chunking, and store the "
        "resulting vectors in the retrieval index."
    ),
    responses=DOCUMENT_ERROR_RESPONSES,
)
async def upload_documents(
    files: UploadFiles,
) -> list[DocumentIngestionResponse]:
    """Upload documents and ingest them into the retrieval stack."""
    try:
        files = await validate_document_upload_files(files)
        return await handle_knowledge_base_ingestion(files)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document upload endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload and ingest documents.",
        ) from exc


@router.post(
    "/knowledge-base/ingest",
    response_model=list[DocumentIngestionResponse],
    status_code=status.HTTP_200_OK,
    deprecated=True,
    summary="Legacy upload-and-ingest alias",
    description=(
        "Deprecated alias for `/upload`. Uploads documents, runs OCR and chunking, "
        "and stores the resulting vectors in the retrieval index."
    ),
    responses=DOCUMENT_ERROR_RESPONSES,
)
async def ingest_knowledge_base_documents(
    files: UploadFiles,
) -> list[DocumentIngestionResponse]:
    """Legacy batch ingestion endpoint for retrieval documents."""
    try:
        files = await validate_document_upload_files(files)
        return await handle_knowledge_base_ingestion(files)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Knowledge-base ingestion endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest knowledge-base documents.",
        ) from exc
