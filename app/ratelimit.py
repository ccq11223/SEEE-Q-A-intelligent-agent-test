"""极简限流：每个 IP 每小时最多提 N 个问题。

存在的理由是省钱：一次误触（比如回车键卡住）就可能在几秒内发出几十个问题，
每个都要调三次模型。进程内内存计数足够应对试点规模，重启即清零。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class HourlyLimiter:
    def __init__(self, limit: int, window_s: float = 3600.0) -> None:
        self.limit = max(1, limit)
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """记一次并返回是否放行。超限时不记录，避免把窗口越推越远。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits[key]
            cutoff = moment - self.window_s
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(moment)
            return True

    def remaining(self, key: str, *, now: float | None = None) -> int:
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits[key]
            cutoff = moment - self.window_s
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            return max(0, self.limit - len(bucket))
