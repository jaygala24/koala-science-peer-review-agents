from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Paper, PublicAgent


class TransparencyLogger:
    def __init__(self, logs_dir: Path, github_repo_url: str, github_branch: str) -> None:
        self.logs_dir = logs_dir
        self.github_repo_url = github_repo_url.rstrip("/")
        self.github_branch = github_branch
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def write_log(
        self,
        *,
        kind: str,
        agent: PublicAgent,
        paper: Paper,
        content_markdown: str,
        metadata: dict[str, Any],
    ) -> tuple[Path, str]:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        paper_slug = slugify(paper.id or paper.title or "paper")
        local_path = self.logs_dir / f"agent_{agent.slot}" / paper_slug / f"{stamp}_{kind}.md"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            f"# {kind.title()} Log",
            "",
            f"- agent: {agent.name}",
            f"- public_role: {agent.public_role}",
            f"- paper_id: {paper.id}",
            f"- paper_title: {paper.title}",
            f"- timestamp_utc: {stamp}",
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(to_jsonable(metadata), indent=2, sort_keys=True),
            "```",
            "",
            "## Public Content",
            "",
            content_markdown,
            "",
        ]
        local_path.write_text("\n".join(body), encoding="utf-8")
        return local_path, self.github_url_for(local_path)

    def github_url_for(self, local_path: Path) -> str:
        if not self.github_repo_url:
            return "https://github.com/your-org/your-agent-repo/blob/main/REPLACE_ME.md"
        relative = local_path.as_posix()
        return f"{self.github_repo_url}/blob/{self.github_branch}/{relative}"


class TrajectoryLogger:
    def __init__(self, logs_dir: Path) -> None:
        self.root = logs_dir / "trajectories"
        self.root.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        agent: PublicAgent | None,
        event: str,
        paper: Paper | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        slot = agent.slot if agent else 0
        name = agent.name if agent else "system"
        role = agent.public_role if agent else "system"
        stamp = datetime.now(tz=UTC).isoformat()
        event_payload = {
            "ts": stamp,
            "event": event,
            "agent_slot": slot,
            "agent_name": name,
            "public_role": role,
            "paper_id": paper.id if paper else None,
            "paper_title": paper.title if paper else None,
            "payload": to_jsonable(payload or {}),
        }
        agent_dir = self.root / f"agent_{slot}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        with (agent_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, sort_keys=True) + "\n")
        with (agent_dir / "trajectory.md").open("a", encoding="utf-8") as handle:
            handle.write(format_trajectory_markdown(event_payload))


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return value[:80] or "paper"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def format_trajectory_markdown(event: dict[str, Any]) -> str:
    payload = event.get("payload") or {}
    lines = [
        f"\n## {event['ts']} - {event['event']}",
        "",
        f"- agent: {event['agent_name']}",
        f"- public_role: {event['public_role']}",
    ]
    if event.get("paper_id"):
        lines.append(f"- paper_id: {event['paper_id']}")
    if event.get("paper_title"):
        lines.append(f"- paper_title: {event['paper_title']}")
    if payload:
        lines.extend(["", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    else:
        lines.append("")
    return "\n".join(lines)
