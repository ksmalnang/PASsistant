"""Disk-backed parent chunk storage with Redis cache acceleration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.utils.cache import RedisCache, get_cache


class ParentChunkStore:
    """Persist parent chunks to JSON while mirroring hot entries in Redis."""

    def __init__(
        self,
        store_path: Path | None = None,
        cache: RedisCache | None = None,
    ):
        settings = get_settings()
        base_dir = settings.processed_data_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = store_path or (base_dir / "parent_chunks.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = cache or get_cache()

    def put_many(self, parents: list[dict[str, Any]]) -> None:
        """Insert or replace parent chunk records."""
        data = self._read_all()
        for parent in parents:
            data[parent["parent_id"]] = parent
            self.cache.set_json(self._parent_key(parent["parent_id"]), parent, ttl_seconds=0)
            self.cache.add_to_set(
                self._document_key(parent["document_id"]),
                parent["parent_id"],
                ttl_seconds=0,
            )
        self._write_all(data)

    def get(self, parent_id: str) -> dict[str, Any] | None:
        """Load a single parent chunk by its identifier."""
        cached = self.cache.get_json(self._parent_key(parent_id))
        if cached is not None:
            return cached

        parent = self._read_all().get(parent_id)
        if parent is not None:
            self.cache.set_json(self._parent_key(parent_id), parent, ttl_seconds=0)
        return parent

    def delete_document(self, document_id: str) -> None:
        """Delete all parent chunks belonging to a document."""
        data = self._read_all()
        stored_parent_ids = [
            key
            for key, value in data.items()
            if value.get("document_id") == document_id
        ]
        filtered = {
            key: value
            for key, value in data.items()
            if value.get("document_id") != document_id
        }
        self._write_all(filtered)
        cached_parent_ids = self.cache.get_set_members(self._document_key(document_id))
        parent_ids = sorted(set(stored_parent_ids) | set(cached_parent_ids))
        self.cache.delete_many(
            [self._parent_key(parent_id) for parent_id in parent_ids]
            + [self._document_key(document_id)]
        )

    def _read_all(self) -> dict[str, dict[str, Any]]:
        """Read the entire store from disk."""
        if not self.store_path.exists():
            return {}
        raw = self.store_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        """Persist the current store state to disk."""
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _parent_key(self, parent_id: str) -> str:
        """Return the cache key for a parent chunk."""
        return f"parent_chunk:{parent_id}"

    def _document_key(self, document_id: str) -> str:
        """Return the cache key for a document-to-parent index."""
        return f"document_parents:{document_id}"
