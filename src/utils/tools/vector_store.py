"""Legacy import shim for :class:`VectorStoreTools`.

The actual implementation lives under ``src.utils.vector_store``. This module
exists only so older imports from ``src.utils.tools.vector_store`` keep working.
"""

from src.utils.vector_store import VectorStoreTools

__all__ = ["VectorStoreTools"]
