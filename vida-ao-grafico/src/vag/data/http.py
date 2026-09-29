"""HTTP com retry/backoff para as fontes públicas de dados."""

from __future__ import annotations

import time

import requests

_session = requests.Session()
_session.headers["User-Agent"] = "vida-ao-grafico/0.1 (research)"


class NotFound(Exception):
    pass


def get(url: str, retries: int = 5, timeout: float = 60, stream: bool = False) -> requests.Response:
    """GET com backoff exponencial. 404 levanta NotFound (arquivo não existe na fonte)."""
    delay = 2.0
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout, stream=stream)
            if r.status_code == 404:
                raise NotFound(url)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {url}")
            r.raise_for_status()
            return r
        except NotFound:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")
