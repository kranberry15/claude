"""Polite HTTP for hkjc.com: one shared rate limit, retries, on-disk cache."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (personal research; hkjc-overlay-tracker)"
_last = [0.0]


class Fetcher:
    def __init__(self, min_interval: float = 1.0, cache_dir: Path | None = None, retries: int = 3, timeout: float = 30):
        self.min_interval = min_interval
        self.cache_dir = cache_dir
        self.retries = retries
        self.timeout = timeout

    def _wait(self):
        dt = time.monotonic() - _last[0]
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        _last[0] = time.monotonic()

    def _request(self, req: urllib.request.Request) -> bytes:
        err = None
        for attempt in range(self.retries):
            self._wait()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError) as e:
                err = e
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 403, 404):
                    break
                time.sleep(2 ** attempt)
        raise err

    def get_text(self, url: str, use_cache: bool = True) -> str:
        path = None
        if self.cache_dir and use_cache:
            path = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")
            if path.exists():
                return path.read_text(encoding="utf-8")
        body = self._request(urllib.request.Request(url, headers={"User-Agent": UA})).decode("utf-8", "replace")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        return body

    def post_json(self, url: str, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"User-Agent": UA, "Content-Type": "application/json", "Accept": "application/json", **headers},
        )
        return json.loads(self._request(req))
