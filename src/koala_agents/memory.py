from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import BayesianUpdate, PublicAgent, RoleResult


class MemoryStore:
    def __init__(self, path: Path, logs_dir: Path) -> None:
        self.path = path
        self.logs_dir = logs_dir / "memory"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists calibration_memory (
                    id integer primary key autoincrement,
                    created_at text not null,
                    agent_slot integer not null,
                    paper_id text not null,
                    prior_probability real,
                    posterior_probability real,
                    confidence real,
                    score real,
                    signal_count integer,
                    payload_json text not null
                );

                create table if not exists comment_roi_memory (
                    id integer primary key autoincrement,
                    created_at text not null,
                    agent_slot integer not null,
                    paper_id text not null,
                    action text not null,
                    ev real,
                    public_role text,
                    payload_json text not null
                );

                create table if not exists domain_memory (
                    domain text not null,
                    agent_slot integer not null,
                    observations integer not null default 0,
                    avg_confidence real not null default 0,
                    avg_posterior real not null default 0,
                    updated_at text not null,
                    primary key (domain, agent_slot)
                );

                create table if not exists external_agent_memory (
                    author_id text primary key,
                    observations integer not null default 0,
                    avg_signal_weight real not null default 0,
                    avg_evidence_quality real not null default 0,
                    avg_confidence real not null default 0,
                    updated_at text not null
                );

                create table if not exists retrieval_memory (
                    query text primary key,
                    results_json text not null,
                    updated_at text not null
                );

                create table if not exists failure_memory (
                    id integer primary key autoincrement,
                    created_at text not null,
                    agent_slot integer,
                    paper_id text,
                    category text not null,
                    message text not null,
                    payload_json text not null
                );
                """
            )

    def record_comment_action(self, agent: PublicAgent, paper_id: str, action: str, ev: float, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into comment_roi_memory (created_at, agent_slot, paper_id, action, ev, public_role, payload_json)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (now_iso(), agent.slot, paper_id, action, ev, agent.public_role, json.dumps(to_jsonable(payload))),
            )

    def record_verdict(
        self,
        agent: PublicAgent,
        paper_id: str,
        score: float,
        update: BayesianUpdate,
        role_results: list[RoleResult],
        domains: list[str],
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into calibration_memory (
                    created_at, agent_slot, paper_id, prior_probability, posterior_probability,
                    confidence, score, signal_count, payload_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso(),
                    agent.slot,
                    paper_id,
                    update.prior_probability,
                    update.posterior_probability,
                    update.posterior_confidence,
                    score,
                    update.signal_count,
                    json.dumps(to_jsonable({"role_results": role_results, **payload})),
                ),
            )
            for domain in domains or ["unknown"]:
                self._update_domain(conn, domain, agent.slot, update.posterior_confidence, update.posterior_probability)

    def record_external_signals(self, signals: list[Any]) -> None:
        with self.connect() as conn:
            for signal in signals:
                weight = 0.35 * signal.confidence + 0.30 * signal.evidence_quality + 0.20 * signal.independence + 0.15 * signal.relevance
                row = conn.execute(
                    "select observations, avg_signal_weight, avg_evidence_quality, avg_confidence from external_agent_memory where author_id = ?",
                    (signal.author_id,),
                ).fetchone()
                if row:
                    n = int(row["observations"])
                    conn.execute(
                        """
                        update external_agent_memory set
                            observations = ?,
                            avg_signal_weight = ?,
                            avg_evidence_quality = ?,
                            avg_confidence = ?,
                            updated_at = ?
                        where author_id = ?
                        """,
                        (
                            n + 1,
                            rolling_avg(float(row["avg_signal_weight"]), weight, n),
                            rolling_avg(float(row["avg_evidence_quality"]), signal.evidence_quality, n),
                            rolling_avg(float(row["avg_confidence"]), signal.confidence, n),
                            now_iso(),
                            signal.author_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        insert into external_agent_memory (
                            author_id, observations, avg_signal_weight, avg_evidence_quality,
                            avg_confidence, updated_at
                        ) values (?, 1, ?, ?, ?, ?)
                        """,
                        (signal.author_id, weight, signal.evidence_quality, signal.confidence, now_iso()),
                    )

    def record_retrieval(self, query: str, results: list[Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into retrieval_memory (query, results_json, updated_at)
                values (?, ?, ?)
                on conflict(query) do update set
                    results_json = excluded.results_json,
                    updated_at = excluded.updated_at
                """,
                (query, json.dumps(to_jsonable(results)), now_iso()),
            )

    def record_failure(self, category: str, message: str, *, agent: PublicAgent | None = None, paper_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into failure_memory (created_at, agent_slot, paper_id, category, message, payload_json)
                values (?, ?, ?, ?, ?, ?)
                """,
                (now_iso(), agent.slot if agent else None, paper_id, category, message, json.dumps(to_jsonable(payload or {}))),
            )

    def export_markdown(self) -> None:
        self._export_table("calibration_memory", self.logs_dir / "calibration.md")
        self._export_table("comment_roi_memory", self.logs_dir / "comment_roi.md")
        self._export_table("domain_memory", self.logs_dir / "domains.md")
        self._export_table("external_agent_memory", self.logs_dir / "external_agents.md")
        self._export_table("failure_memory", self.logs_dir / "failures.md")

    def _update_domain(self, conn: sqlite3.Connection, domain: str, agent_slot: int, confidence: float, posterior: float) -> None:
        row = conn.execute(
            "select observations, avg_confidence, avg_posterior from domain_memory where domain = ? and agent_slot = ?",
            (domain, agent_slot),
        ).fetchone()
        if row:
            n = int(row["observations"])
            conn.execute(
                """
                update domain_memory set observations = ?, avg_confidence = ?, avg_posterior = ?, updated_at = ?
                where domain = ? and agent_slot = ?
                """,
                (
                    n + 1,
                    rolling_avg(float(row["avg_confidence"]), confidence, n),
                    rolling_avg(float(row["avg_posterior"]), posterior, n),
                    now_iso(),
                    domain,
                    agent_slot,
                ),
            )
        else:
            conn.execute(
                """
                insert into domain_memory (domain, agent_slot, observations, avg_confidence, avg_posterior, updated_at)
                values (?, ?, 1, ?, ?, ?)
                """,
                (domain, agent_slot, confidence, posterior, now_iso()),
            )

    def _export_table(self, table: str, path: Path) -> None:
        with self.connect() as conn:
            rows = conn.execute(f"select * from {table} order by rowid desc limit 200").fetchall()
        lines = [f"# {table}", ""]
        for row in rows:
            lines.extend(["## Entry", "", "```json", json.dumps(dict(row), indent=2, sort_keys=True), "```", ""])
        path.write_text("\n".join(lines), encoding="utf-8")


def rolling_avg(current: float, value: float, n: int) -> float:
    return ((current * n) + value) / (n + 1)


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


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
