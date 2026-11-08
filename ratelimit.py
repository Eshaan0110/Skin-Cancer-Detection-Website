"""Small in-memory sliding-window rate limiter keyed by client address."""
import math
import os
import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
DEFAULT_LIMIT = 30

_lock = threading.Lock()
_hits = defaultdict(deque)


def configured_limit():
    """Images allowed per client per minute; 0 or negative disables limiting."""
    raw = os.environ.get("RATE_LIMIT_PER_MINUTE", "")
    try:
        return int(raw) if raw.strip() else DEFAULT_LIMIT
    except ValueError:
        return DEFAULT_LIMIT


def check(client, cost=1, now=None):
    """Record `cost` images for `client`.

    Returns (allowed, retry_after_seconds). A request whose cost exceeds the
    remaining budget is rejected without being recorded.
    """
    limit = configured_limit()
    if limit <= 0:
        return True, 0
    now = time.monotonic() if now is None else now
    cutoff = now - WINDOW_SECONDS
    with _lock:
        q = _hits[client]
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) + cost > limit:
            if cost > limit or not q:
                return False, WINDOW_SECONDS
            # Wait until enough old hits expire to fit this request.
            idx = len(q) + cost - limit - 1
            return False, max(1, math.ceil(q[idx] + WINDOW_SECONDS - now))
        q.extend([now] * cost)
    return True, 0


def reset():
    with _lock:
        _hits.clear()
