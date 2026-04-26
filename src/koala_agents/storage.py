from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ActionDecision


class AgentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists actions (
                    id integer primary key autoincrement,
                    created_at text not null,
                    agent_slot integer not null,
                    agent_name text not null,
                    paper_id text not null,
                    action text not null,
                    ev real not null,
                    reason text not null,
                    roles_json text not null,
                    features_json text not null,
                    parent_comment_id text,
                    dry_run integer not null,
                    response_json text
                );

                create table if not exists agent_papers (
                    agent_slot integer not null,
                    paper_id text not null,
                    first_comment_id text,
                    verdict_posted integer not null default 0,
                    updated_at text not null,
                    primary key (agent_slot, paper_id)
                );

                create table if not exists comments (
                    comment_id text primary key,
                    paper_id text not null,
                    agent_slot integer not null,
                    parent_comment_id text,
                    content_markdown text not null,
                    github_file_url text,
                    created_at text not null
                );

                create table if not exists verdicts (
                    paper_id text not null,
                    agent_slot integer not null,
                    score real not null,
                    content_markdown text not null,
                    github_file_url text,
                    created_at text not null,
                    primary key (paper_id, agent_slot)
                );
                """
            )

    def record_decision(
        self,
        decision: ActionDecision,
        *,
        dry_run: bool,
        response: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into actions (
                    created_at, agent_slot, agent_name, paper_id, action, ev, reason,
                    roles_json, features_json, parent_comment_id, dry_run, response_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso(),
                    decision.agent.slot,
                    decision.agent.name,
                    decision.paper.id,
                    decision.action,
                    decision.ev,
                    decision.reason,
                    json.dumps(decision.roles),
                    json.dumps(asdict(decision.features)),
                    decision.parent_comment_id,
                    int(dry_run),
                    json.dumps(response or {}),
                ),
            )

    def agent_has_commented(self, agent_slot: int, paper_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "select 1 from agent_papers where agent_slot = ? and paper_id = ? and first_comment_id is not null",
                (agent_slot, paper_id),
            ).fetchone()
        return row is not None

    def touched_agent_slots(self, paper_id: str) -> set[int]:
        with self.connect() as conn:
            rows = conn.execute(
                "select agent_slot from agent_papers where paper_id = ?", (paper_id,)
            ).fetchall()
        return {int(row["agent_slot"]) for row in rows}

    def mark_comment(
        self,
        *,
        agent_slot: int,
        paper_id: str,
        comment_id: str,
        content_markdown: str,
        github_file_url: str,
        parent_comment_id: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into comments (
                    comment_id, paper_id, agent_slot, parent_comment_id, content_markdown,
                    github_file_url, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comment_id,
                    paper_id,
                    agent_slot,
                    parent_comment_id,
                    content_markdown,
                    github_file_url,
                    now_iso(),
                ),
            )
            conn.execute(
                """
                insert into agent_papers (agent_slot, paper_id, first_comment_id, updated_at)
                values (?, ?, ?, ?)
                on conflict(agent_slot, paper_id) do update set
                    first_comment_id = coalesce(agent_papers.first_comment_id, excluded.first_comment_id),
                    updated_at = excluded.updated_at
                """,
                (agent_slot, paper_id, comment_id, now_iso()),
            )

    def mark_verdict(
        self,
        *,
        agent_slot: int,
        paper_id: str,
        score: float,
        content_markdown: str,
        github_file_url: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into verdicts (
                    paper_id, agent_slot, score, content_markdown, github_file_url, created_at
                ) values (?, ?, ?, ?, ?, ?)
                """,
                (paper_id, agent_slot, score, content_markdown, github_file_url, now_iso()),
            )
            conn.execute(
                """
                insert into agent_papers (agent_slot, paper_id, verdict_posted, updated_at)
                values (?, ?, 1, ?)
                on conflict(agent_slot, paper_id) do update set
                    verdict_posted = 1,
                    updated_at = excluded.updated_at
                """,
                (agent_slot, paper_id, now_iso()),
            )

    def verdict_posted(self, agent_slot: int, paper_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "select 1 from verdicts where agent_slot = ? and paper_id = ?",
                (agent_slot, paper_id),
            ).fetchone()
        return row is not None

    def paid_comment_karma_spent(
        self,
        agent_slot: int,
        *,
        since: datetime | None = None,
        live_only: bool = True,
    ) -> float:
        clauses = ["agent_slot = ?", "action in ('first_comment', 'reply', 'follow_up')"]
        params: list[Any] = [agent_slot]
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        if live_only:
            clauses.append("dry_run = 0")
        where = " and ".join(clauses)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                select coalesce(sum(
                    case action
                        when 'reply' then 0.1
                        else 1.0
                    end
                ), 0.0) as spent
                from actions
                where {where}
                """,
                params,
            ).fetchone()
        return float(row["spent"] if row else 0.0)


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
