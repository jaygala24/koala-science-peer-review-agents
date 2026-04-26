from __future__ import annotations

from typing import Any

from ..models import Comment, Paper, PublicAgent
from .http import request_json


class KoalaClient:
    def __init__(self, api_base: str, agent: PublicAgent | None = None) -> None:
        self.api_base = api_base.rstrip("/")
        self.agent = agent

    def with_agent(self, agent: PublicAgent) -> "KoalaClient":
        return KoalaClient(self.api_base, agent)

    @property
    def headers(self) -> dict[str, str]:
        if not self.agent or not self.agent.api_key:
            return {}
        return {"Authorization": self.agent.api_key}

    def get_my_profile(self) -> dict[str, Any]:
        return request_json("GET", f"{self.api_base}/users/me", headers=self.headers)

    def update_my_profile(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        github_repo: str | None = None,
    ) -> dict[str, Any]:
        body = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "github_repo": github_repo,
            }.items()
            if value
        }
        return request_json("PATCH", f"{self.api_base}/users/me", headers=self.headers, body=body)

    def get_papers(self, *, domain: str | None = None, limit: int = 20) -> list[Paper]:
        data = request_json(
            "GET",
            f"{self.api_base}/papers/",
            headers=self.headers,
            query={"domain": domain, "limit": limit},
        )
        return [Paper.from_api(item) for item in normalize_list(data)]

    def search_papers(
        self,
        query: str,
        *,
        domain: str | None = None,
        result_type: str = "all",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        data = request_json(
            "GET",
            f"{self.api_base}/search/",
            headers=self.headers,
            query={"q": query, "domain": domain, "type": result_type, "limit": limit},
        )
        return normalize_list(data)

    def get_paper(self, paper_id: str) -> Paper:
        data = request_json("GET", f"{self.api_base}/papers/{paper_id}", headers=self.headers)
        return Paper.from_api(data)

    def get_comments(self, paper_id: str, *, limit: int = 100) -> list[Comment]:
        data = request_json(
            "GET",
            f"{self.api_base}/comments/paper/{paper_id}",
            headers=self.headers,
            query={"limit": limit},
        )
        return [Comment.from_api(item) for item in normalize_list(data)]

    def post_comment(
        self,
        paper_id: str,
        content_markdown: str,
        github_file_url: str,
        *,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "paper_id": paper_id,
            "content_markdown": content_markdown,
            "github_file_url": github_file_url,
        }
        if parent_id:
            body["parent_id"] = parent_id
        return request_json("POST", f"{self.api_base}/comments/", headers=self.headers, body=body)

    def post_verdict(
        self,
        paper_id: str,
        content_markdown: str,
        score: float,
        github_file_url: str,
        *,
        flagged_agent_id: str | None = None,
        flag_reason: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "paper_id": paper_id,
            "content_markdown": content_markdown,
            "score": score,
            "github_file_url": github_file_url,
        }
        if flagged_agent_id and flag_reason:
            body["flagged_agent_id"] = flagged_agent_id
            body["flag_reason"] = flag_reason
        return request_json("POST", f"{self.api_base}/verdicts/", headers=self.headers, body=body)


def normalize_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "items", "papers", "comments", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []
