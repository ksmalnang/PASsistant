"""
FastAPI REST API for the academic services and student records chatbot.

Provides HTTP endpoints for:
- Chat interactions (WebSocket and REST)
- Document ingestion and retrieval support
- Health checks

Usage:
    uvicorn src.api:app --reload --port 8000
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.api.routes.router import router
from src.config import build_logging_config, configure_logging, get_settings
from src.guardrails.rate_limit import InMemoryRateLimiter

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting academic services and student records chatbot API")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Academic Services and Student Records Chatbot API",
    description=(
        "Production-ready LangGraph chatbot API for academic-service questions, "
        "chat interactions, and document ingestion backed by retrieval indexing"
    ),
    version="0.1.0",
    openapi_version="3.0.3",
    lifespan=lifespan,
)

allowed_origins = settings.CORS_ALLOWED_ORIGINS
if settings.is_production and allowed_origins == ["*"]:
    allowed_origins = []

app.state.rate_limiter = InMemoryRateLimiter(limit=settings.RATE_LIMIT_PER_MINUTE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router)


def _patch_binary_upload_schemas(node):
    """Convert OpenAPI 3.1 contentMediaType upload schemas into Swagger-friendly binary schemas."""
    if isinstance(node, dict):
        if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType", None)
            node["format"] = "binary"
        for value in node.values():
            _patch_binary_upload_schemas(value)
    elif isinstance(node, list):
        for item in node:
            _patch_binary_upload_schemas(item)


def custom_openapi():
    """Build and patch the OpenAPI schema for Swagger file upload compatibility."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"
    _patch_binary_upload_schemas(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


def main() -> None:
    """Run the FastAPI application with uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=build_logging_config(settings),
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
