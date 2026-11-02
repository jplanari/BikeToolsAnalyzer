"""Small security helpers shared by the phone API and the desktop login."""
import threading
import time


class Throttle:
    """Slows down guessing: after `limit` failures under one key (an address, a rider name) within `window`
    seconds, that key is refused until the oldest failure ages out."""
    def __init__(self, limit=10, window=60):
        self.limit, self.window = limit, window
        self._fails = {}
        self._lock = threading.Lock()

    def _recent(self, key, now):
        hits = [t for t in self._fails.get(key, []) if now - t < self.window]
        if hits:
            self._fails[key] = hits
        else:
            self._fails.pop(key, None)
        return hits

    def blocked(self, key, now=None):
        with self._lock:
            return len(self._recent(key, now or time.time())) >= self.limit

    def fail(self, key, now=None):
        now = now or time.time()
        with self._lock:
            self._recent(key, now)
            self._fails.setdefault(key, []).append(now)
            if len(self._fails) > 5000:         # bound the memory a scan from many addresses could use
                self._fails.clear()

    def retry_after(self, key, now=None):
        """Seconds until `key` may try again (0 when it may now)."""
        now = now or time.time()
        with self._lock:
            hits = self._recent(key, now)
            return 0 if len(hits) < self.limit else max(1, int(self.window - (now - hits[0])) + 1)

    def reset(self, key):
        with self._lock:
            self._fails.pop(key, None)
