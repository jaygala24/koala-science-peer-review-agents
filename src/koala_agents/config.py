from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import PublicAgent


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    koala_api_base: str
    github_repo_url: str
    github_branch: str
    gemini_api_key: str
    gemini_model: str
    semantic_scholar_api_key: str
    db_path: Path
    memory_db_path: Path
    logs_dir: Path
    public_logs_dir: Path
    dry_run: bool
    exhaustive_mode: bool
    max_internal_roles: int
    enable_retrieval_for_all_agents: bool
    min_verdict_citations: int
    max_actions_per_run: int
    use_gemini_in_dry_run: bool
    enable_model_tool_calls: bool
    max_tool_call_rounds: int
    enable_dynamic_pacing: bool
    max_comment_karma_per_agent_per_day: float
    comment_karma_budget_per_agent: float
    pacing_lookback_hours: float
    bayesian_discussion_updates: bool
    max_discussion_update_weight: float
    loop_interval_seconds: int
    agents: list[PublicAgent]

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "AppConfig":
        load_env_file(env_path)
        agents: list[PublicAgent] = []
        defaults = {
            1: ("EvidenceRigorAgent", "evidence_rigor", 35.0),
            2: ("LiteratureNoveltyAgent", "literature_novelty", 35.0),
            3: ("MetaCalibrationAgent", "meta_calibration", 40.0),
        }
        for slot, (default_name, default_role, default_reserve) in defaults.items():
            agents.append(
                PublicAgent(
                    slot=slot,
                    name=os.environ.get(f"KOALA_AGENT_{slot}_NAME", default_name),
                    api_key=os.environ.get(f"KOALA_AGENT_{slot}_API_KEY", ""),
                    public_role=os.environ.get(f"KOALA_AGENT_{slot}_PUBLIC_ROLE", default_role),
                    min_karma_reserve=env_float(
                        f"KOALA_AGENT_{slot}_MIN_KARMA_RESERVE", default_reserve
                    ),
                )
            )

        return cls(
            koala_api_base=os.environ.get("KOALA_API_BASE", "https://koala.science/api/v1").rstrip(
                "/"
            ),
            github_repo_url=os.environ.get("GITHUB_REPO_URL", "").rstrip("/"),
            github_branch=os.environ.get("GITHUB_BRANCH", "main"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
            db_path=Path(os.environ.get("DB_PATH", "data/koala_agents.sqlite3")),
            memory_db_path=Path(os.environ.get("MEMORY_DB_PATH", "data/memory.sqlite3")),
            logs_dir=Path(os.environ.get("LOGS_DIR", "logs")),
            public_logs_dir=Path(os.environ.get("PUBLIC_LOGS_DIR", "public_logs")),
            dry_run=env_bool("DRY_RUN", True),
            exhaustive_mode=env_bool("EXHAUSTIVE_MODE", True),
            max_internal_roles=int(env_float("MAX_INTERNAL_ROLES", 12.0)),
            enable_retrieval_for_all_agents=env_bool("ENABLE_RETRIEVAL_FOR_ALL_AGENTS", True),
            min_verdict_citations=int(env_float("MIN_VERDICT_CITATIONS", 3.0)),
            max_actions_per_run=int(env_float("MAX_ACTIONS_PER_RUN", 3.0)),
            use_gemini_in_dry_run=env_bool("USE_GEMINI_IN_DRY_RUN", True),
            enable_model_tool_calls=env_bool("ENABLE_MODEL_TOOL_CALLS", True),
            max_tool_call_rounds=int(env_float("MAX_TOOL_CALL_ROUNDS", 2.0)),
            enable_dynamic_pacing=env_bool("ENABLE_DYNAMIC_PACING", True),
            max_comment_karma_per_agent_per_day=env_float("MAX_COMMENT_KARMA_PER_AGENT_PER_DAY", 8.0),
            comment_karma_budget_per_agent=env_float("COMMENT_KARMA_BUDGET_PER_AGENT", 55.0),
            pacing_lookback_hours=env_float("PACING_LOOKBACK_HOURS", 24.0),
            bayesian_discussion_updates=env_bool("BAYESIAN_DISCUSSION_UPDATES", True),
            max_discussion_update_weight=env_float("MAX_DISCUSSION_UPDATE_WEIGHT", 0.35),
            loop_interval_seconds=int(env_float("LOOP_INTERVAL_SECONDS", 300.0)),
            agents=agents,
        )

    @property
    def active_agents(self) -> list[PublicAgent]:
        if self.dry_run:
            return self.agents
        return [agent for agent in self.agents if agent.enabled]
