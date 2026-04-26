from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

from ..models import RetrievalResult
from .http import ApiError, request_json


class QueryGuard:
    """Conservative guard against identity-leaking retrieval queries."""

    _authorish = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")

    def sanitize(self, query: str) -> str:
        query = " ".join(query.split())
        if len(query) > 180:
            query = query[:180]
        return query

    def is_safe(self, query: str) -> bool:
        lowered = query.lower()
        blocked = ["openreview", "submitted to icml", "anonymous authors"]
        if any(term in lowered for term in blocked):
            return False
        if len(query) > 220:
            return False
        if re.search(r"\b(author|authors|by|from)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b", query):
            return False
        tokens = query.split()
        if len(tokens) <= 3 and self._authorish.search(query):
            return False
        return True


class SemanticScholarClient:
    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self.guard = QueryGuard()

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        query = self.guard.sanitize(query)
        if not self.guard.is_safe(query):
            return []
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        try:
            data = request_json(
                "GET",
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                query={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,year,venue,url,externalIds",
                },
                retries=1,
            )
        except ApiError:
            return []
        items = data.get("data", []) if isinstance(data, dict) else []
        return [self._to_result(item) for item in items if isinstance(item, dict)]

    def _to_result(self, item: dict[str, Any]) -> RetrievalResult:
        url = item.get("url")
        external_ids = item.get("externalIds") or {}
        if external_ids.get("ArXiv"):
            url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
        return RetrievalResult(
            title=str(item.get("title") or ""),
            source="semantic_scholar",
            url=url,
            year=item.get("year"),
            venue=item.get("venue"),
            abstract=item.get("abstract"),
        )


class ArxivClient:
    def __init__(self) -> None:
        self.guard = QueryGuard()

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        query = self.guard.sanitize(query)
        if not self.guard.is_safe(query):
            return []
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query)}&start=0&max_results={limit}"
        )
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return self._parse(raw)

    def _parse(self, raw: str) -> list[RetrievalResult]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        results: list[RetrievalResult] = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            url = entry.findtext("atom:id", default=None, namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            results.append(
                RetrievalResult(
                    title=title,
                    source="arxiv",
                    url=url,
                    year=year,
                    venue="arXiv",
                    abstract=summary,
                )
            )
        return results


class RetrievalBroker:
    def __init__(self, semantic_scholar_api_key: str = "") -> None:
        self.semantic_scholar = SemanticScholarClient(semantic_scholar_api_key)
        self.arxiv = ArxivClient()

    def search_many(self, queries: list[str], *, per_query: int = 4) -> list[RetrievalResult]:
        seen: set[str] = set()
        results: list[RetrievalResult] = []
        for query in queries:
            for result in [
                *self.semantic_scholar.search(query, limit=per_query),
                *self.arxiv.search(query, limit=max(1, per_query // 2)),
            ]:
                key = (result.title or result.url or "").lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(result)
        return results
