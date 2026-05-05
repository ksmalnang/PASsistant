"""Simple in-memory rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Enforce a fixed number of requests per rolling window."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = max(limit, 1)
        self._window_seconds = max(window_seconds, 1)
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """Return whether the key is allowed to proceed."""
        now = time.monotonic()
        events = self._events[key]
        cutoff = now - self._window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self._limit:
            return False
        events.append(now)
        return True
