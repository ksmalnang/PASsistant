"""Composed API router."""

from fastapi import APIRouter

from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.api.routes.websocket import router as websocket_router

router = APIRouter()

router.include_router(health_router)
router.include_router(chat_router)
router.include_router(documents_router)
router.include_router(websocket_router)
