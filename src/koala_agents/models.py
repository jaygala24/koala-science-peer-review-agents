from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


PaperStatus = Literal["in_review", "deliberating", "reviewed", "unknown"]
ActionType = Literal["skip", "observe", "first_comment", "reply", "follow_up", "verdict"]


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return None


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    status: PaperStatus = "unknown"
    domains: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    pdf_url: str | None = None
    tarball_url: str | None = None
    github_urls: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Paper":
        paper_id = str(data.get("id") or data.get("paper_id") or data.get("uuid") or "")
        github_urls = data.get("github_urls") or []
        if data.get("github_repo_url") and data.get("github_repo_url") not in github_urls:
            github_urls = [*github_urls, data["github_repo_url"]]
        domains = data.get("domains") or []
        if domains and isinstance(domains[0], dict):
            domains = [str(item.get("name", item.get("id", ""))) for item in domains]
        return cls(
            id=paper_id,
            title=str(data.get("title") or ""),
            abstract=str(data.get("abstract") or ""),
            status=data.get("status", "unknown"),
            domains=[str(domain) for domain in domains if domain],
            created_at=parse_time(data.get("created_at") or data.get("submitted_at")),
            pdf_url=data.get("pdf_url"),
            tarball_url=data.get("tarball_url"),
            github_urls=[str(url) for url in github_urls if url],
            raw=data,
        )


@dataclass(frozen=True)
class Comment:
    id: str
    paper_id: str
    author_id: str
    content_markdown: str
    parent_id: str | None = None
    author_type: str = "unknown"
    created_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Comment":
        return cls(
            id=str(data.get("id") or data.get("comment_id") or ""),
            paper_id=str(data.get("paper_id") or ""),
            author_id=str(data.get("author_id") or data.get("user_id") or ""),
            author_type=str(data.get("author_type") or "unknown"),
            parent_id=data.get("parent_id"),
            content_markdown=str(data.get("content_markdown") or data.get("content") or ""),
            created_at=parse_time(data.get("created_at")),
            raw=data,
        )


@dataclass(frozen=True)
class PublicAgent:
    slot: int
    name: str
    api_key: str
    public_role: str
    min_karma_reserve: float
    actor_id: str | None = None
    karma: float | None = None
    strike_count: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class RoleResult:
    role_name: str
    accept_probability: float
    confidence: float
    strengths: list[str]
    weaknesses: list[str]
    uncertainties: list[str]
    public_points: list[str]
    fatal_flaws: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    score_factors: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""


@dataclass(frozen=True)
class DiscussionSignal:
    comment_id: str
    author_id: str
    stance_probability: float
    confidence: float
    evidence_quality: float
    independence: float
    relevance: float
    summary: str
    useful: bool = True


@dataclass(frozen=True)
class BayesianUpdate:
    prior_probability: float
    prior_confidence: float
    posterior_probability: float
    posterior_confidence: float
    discussion_weight: float
    signal_count: int
    weighted_signal_probability: float | None
    rationale: str


@dataclass(frozen=True)
class RetrievalResult:
    title: str
    source: str
    url: str | None = None
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None


@dataclass(frozen=True)
class OpportunityFeatures:
    verdict_feasibility: float
    domain_fit: float
    prediction_edge: float
    under_reviewed_bonus: float
    discussion_quality: float
    citation_probability: float
    karma_recovery_potential: float
    sibling_overlap_penalty: float
    time_risk: float
    crowding_penalty: float
    redundancy_penalty: float


@dataclass(frozen=True)
class ActionDecision:
    action: ActionType
    agent: PublicAgent
    paper: Paper
    ev: float
    features: OpportunityFeatures
    roles: list[str]
    reason: str
    parent_comment_id: str | None = None
