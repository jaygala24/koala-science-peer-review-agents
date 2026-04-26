from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ApiError(Exception):
    status: int
    message: str
    body: str = ""

    def __str__(self) -> str:
        return f"API error {self.status}: {self.message}"


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout: int = 60,
    retries: int = 2,
) -> Any:
    if query:
        cleaned = {key: value for key, value in query.items() if value is not None and value != ""}
        if cleaned:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(cleaned, doseq=True)}"

    payload = None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    last_error: ApiError | None = None
    for attempt in range(retries + 1):
        req = Request(url, data=payload, headers=request_headers, method=method.upper())
        try:
            with urlopen(req, timeout=timeout) as response:  # noqa: S310 - URLs are configured APIs.
                raw = response.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            last_error = ApiError(status=exc.code, message=exc.reason, body=raw)
            if exc.code not in {429, 503} or attempt >= retries:
                raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
        except URLError as exc:
            last_error = ApiError(status=0, message=str(exc.reason))
            if attempt >= retries:
                raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
    if last_error:
        raise last_error
    raise ApiError(status=0, message="unknown request failure")
