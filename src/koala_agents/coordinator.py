from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from .calibration import aggregate_prior, bayesian_update_from_signals, extract_discussion_signals
from .clients.http import ApiError
from .clients.koala import KoalaClient
from .clients.retrieval import RetrievalBroker
from .config import AppConfig
from .llm.gemini import GeminiRunner
from .logging_utils import TrajectoryLogger, TransparencyLogger
from .memory import MemoryStore
from .models import ActionDecision, Comment, Paper, PublicAgent, RetrievalResult, RoleResult
from .paper_reader import PaperReader
from .roles import retrieval_queries_for
from .storage import AgentStore
from .strategy import DynamicStrategy


class Coordinator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = AgentStore(config.db_path)
        self.logger = TransparencyLogger(config.logs_dir, config.github_repo_url, config.github_branch)
        self.trajectory = TrajectoryLogger(config.logs_dir)
        self.memory = MemoryStore(config.memory_db_path, config.logs_dir)
        self.retrieval = RetrievalBroker(config.semantic_scholar_api_key)
        self.llm = GeminiRunner(
            config.gemini_api_key,
            config.gemini_model,
            dry_run=config.dry_run and not config.use_gemini_in_dry_run,
        )
        self.paper_reader = PaperReader(config.koala_api_base)
        self.agents = self.refresh_agents(config.active_agents)
        self.sibling_actor_ids = {agent.actor_id for agent in self.agents if agent.actor_id}
        self.strategy = DynamicStrategy(
            self.store,
            self.sibling_actor_ids,
            exhaustive=config.exhaustive_mode,
            max_roles=config.max_internal_roles,
        )

    def refresh_agents(self, agents: list[PublicAgent]) -> list[PublicAgent]:
        refreshed: list[PublicAgent] = []
        for agent in agents:
            if not agent.api_key:
                refreshed.append(agent)
                continue
            try:
                profile = KoalaClient(self.config.koala_api_base, agent).get_my_profile()
            except ApiError:
                refreshed.append(agent)
                continue
            refreshed.append(
                replace(
                    agent,
                    actor_id=str(profile.get("id") or profile.get("actor_id") or "") or None,
                    karma=_maybe_float(profile.get("karma")),
                    strike_count=_maybe_int(profile.get("strike_count")),
                )
            )
        return refreshed

    def status(self) -> dict[str, Any]:
        return {
            "dry_run": self.config.dry_run,
            "koala_api_base": self.config.koala_api_base,
            "github_repo_configured": bool(self.config.github_repo_url),
            "gemini_available": self.llm.available(),
            "semantic_scholar_key_configured": bool(self.config.semantic_scholar_api_key),
            "exhaustive_mode": self.config.exhaustive_mode,
            "max_internal_roles": self.config.max_internal_roles,
            "enable_retrieval_for_all_agents": self.config.enable_retrieval_for_all_agents,
            "enable_model_tool_calls": self.config.enable_model_tool_calls,
            "max_tool_call_rounds": self.config.max_tool_call_rounds,
            "min_verdict_citations": self.config.min_verdict_citations,
            "bayesian_discussion_updates": self.config.bayesian_discussion_updates,
            "max_discussion_update_weight": self.config.max_discussion_update_weight,
            "use_gemini_in_dry_run": self.config.use_gemini_in_dry_run,
            "loop_interval_seconds": self.config.loop_interval_seconds,
            "memory_db_path": str(self.config.memory_db_path),
            "agents": [
                {
                    "slot": agent.slot,
                    "name": agent.name,
                    "public_role": agent.public_role,
                    "has_api_key": bool(agent.api_key),
                    "actor_id": agent.actor_id,
                    "karma": agent.karma,
                    "strike_count": agent.strike_count,
                    "min_karma_reserve": agent.min_karma_reserve,
                }
                for agent in self.agents
            ],
        }

    def preflight(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if not self.config.dry_run and not self.config.github_repo_url:
            errors.append("GITHUB_REPO_URL is required before live posting.")
        if "your-org" in self.config.github_repo_url or "your-agent-repo" in self.config.github_repo_url:
            warnings.append("GITHUB_REPO_URL still looks like the placeholder value.")
        live_agents = [agent for agent in self.agents if agent.api_key]
        if not self.config.dry_run and not live_agents:
            errors.append("No Koala agent API keys are configured for live mode.")
        if len(live_agents) < 3:
            warnings.append(f"Only {len(live_agents)} Koala agent API key(s) configured; expected up to 3.")
        if not self.llm.available():
            warnings.append("Gemini is not available; heuristic fallback will be much weaker.")
        if not self.config.semantic_scholar_api_key:
            warnings.append("Semantic Scholar API key is missing; retrieval may be rate-limited.")
        if self.config.max_internal_roles < 6:
            warnings.append("MAX_INTERNAL_ROLES is low for exhaustive review.")
        return {
            "ok": not errors,
            "dry_run": self.config.dry_run,
            "errors": errors,
            "warnings": warnings,
            "status": self.status(),
        }

    def export_memory(self) -> dict[str, Any]:
        self.memory.export_markdown()
        self.trajectory.record(agent=None, event="memory_export", payload={"logs_dir": str(self.config.logs_dir / "memory")})
        return {"exported": True, "path": str(self.config.logs_dir / "memory")}

    def scan(self, *, limit: int = 20, domain: str | None = None) -> list[ActionDecision]:
        papers = self.fetch_papers(limit=limit, domain=domain)
        decisions: list[ActionDecision] = []
        for paper in papers:
            comments = self.safe_comments(paper.id)
            for agent in self.agents:
                self.sync_remote_agent_comments(agent, paper, comments)
                decision = self.strategy.decide(paper, comments, agent)
                decisions.append(decision)
                self.trajectory.record(
                    agent=agent,
                    event="scan_decision",
                    paper=paper,
                    payload={
                        "action": decision.action,
                        "ev": decision.ev,
                        "reason": decision.reason,
                        "roles": decision.roles,
                        "features": decision.features,
                        "comment_count": len(comments),
                    },
                )
        return sorted(decisions, key=lambda decision: decision.ev, reverse=True)

    def run_once(
        self, *, limit: int = 20, domain: str | None = None, max_actions: int | None = None
    ) -> list[dict[str, Any]]:
        max_actions = max_actions or self.config.max_actions_per_run
        candidates = [
            decision
            for decision in self.scan(limit=limit, domain=domain)
            if decision.action in {"first_comment", "reply", "follow_up"}
        ]
        executed: list[dict[str, Any]] = []
        planned_papers: set[tuple[int, str]] = set()
        for decision in candidates:
            key = (decision.agent.slot, decision.paper.id)
            if key in planned_papers:
                continue
            planned_papers.add(key)
            executed.append(self.execute_comment_decision(decision))
            if len(executed) >= max_actions:
                break
        return executed

    def deliberate(
        self, *, limit: int = 50, domain: str | None = None, max_actions: int = 10
    ) -> list[dict[str, Any]]:
        papers = [paper for paper in self.fetch_papers(limit=limit, domain=domain) if paper.status == "deliberating"]
        executed: list[dict[str, Any]] = []
        for paper in papers:
            comments = self.safe_comments(paper.id)
            for agent in self.agents:
                self.sync_remote_agent_comments(agent, paper, comments)
                if not self.store.agent_has_commented(agent.slot, paper.id):
                    continue
                if self.store.verdict_posted(agent.slot, paper.id):
                    continue
                result = self.execute_verdict(agent, paper, comments)
                if result:
                    executed.append(result)
                if len(executed) >= max_actions:
                    return executed
        return executed

    def fetch_papers(self, *, limit: int, domain: str | None) -> list[Paper]:
        client = self.read_client()
        return client.get_papers(domain=domain, limit=limit)

    def safe_comments(self, paper_id: str) -> list[Comment]:
        try:
            return self.read_client().get_comments(paper_id, limit=100)
        except ApiError:
            return []

    def read_client(self) -> KoalaClient:
        keyed = next((agent for agent in self.agents if agent.api_key), None)
        return KoalaClient(self.config.koala_api_base, keyed)

    def update_profiles(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.config.github_repo_url:
            raise RuntimeError("GITHUB_REPO_URL must be configured before updating profiles")
        for agent in self.agents:
            if not agent.api_key:
                results.append({"agent": agent.name, "skipped": True, "reason": "missing API key"})
                continue
            description = profile_description(agent.public_role)
            if self.config.dry_run:
                results.append(
                    {
                        "agent": agent.name,
                        "dry_run": True,
                        "github_repo": self.config.github_repo_url,
                        "description": description,
                    }
                )
                continue
            response = KoalaClient(self.config.koala_api_base, agent).update_my_profile(
                description=description,
                github_repo=self.config.github_repo_url,
            )
            results.append({"agent": agent.name, "response": response})
            self.trajectory.record(agent=agent, event="profile_update", payload={"response": response})
        return results

    def sync_remote_agent_comments(
        self, agent: PublicAgent, paper: Paper, comments: list[Comment]
    ) -> None:
        if not agent.actor_id:
            return
        for comment in comments:
            if comment.author_id != agent.actor_id:
                continue
            self.store.mark_comment(
                agent_slot=agent.slot,
                paper_id=paper.id,
                comment_id=comment.id,
                content_markdown=comment.content_markdown,
                github_file_url=str(comment.raw.get("github_file_url") or ""),
                parent_comment_id=comment.parent_id,
            )

    def execute_comment_decision(self, decision: ActionDecision) -> dict[str, Any]:
        comments = self.safe_comments(decision.paper.id)
        analysis_paper = self.paper_reader.enrich(decision.paper)
        retrieval_results = self.retrieve(decision.paper, decision.agent)
        role_results = self.run_roles(
            replace(decision, paper=analysis_paper), [], retrieval_results, include_discussion=False
        )
        update = self.discussion_update(role_results, comments)
        content = safe_public_markdown(
            self.llm.synthesize_comment(analysis_paper, role_results, decision.agent.public_role)
        )
        content = append_bayesian_note(content, update, include_when_no_signals=decision.action != "first_comment")
        if decision.action == "reply":
            content = "Building on this thread, " + content[0].lower() + content[1:]
            content = safe_public_markdown(content, max_chars=3500)

        metadata = {
            "decision": decision,
            "retrieval_results": retrieval_results,
            "role_results": role_results,
            "bayesian_update": update,
            "discussion_signals": extract_discussion_signals(comments, self.sibling_actor_ids),
            "dry_run": self.config.dry_run,
        }
        _, github_url = self.logger.write_log(
            kind=decision.action,
            agent=decision.agent,
            paper=decision.paper,
            content_markdown=content,
            metadata=metadata,
        )

        if self.config.dry_run:
            response = {"dry_run": True, "would_post": decision.action, "github_file_url": github_url}
            self.store.record_decision(decision, dry_run=True, response=response)
            self.record_action_memory(decision, response, update, role_results, retrieval_results, comments)
            return self.result_payload(decision, response, content)

        self.require_live_ready(decision.agent, github_url)
        response = KoalaClient(self.config.koala_api_base, decision.agent).post_comment(
            decision.paper.id,
            content,
            github_url,
            parent_id=decision.parent_comment_id,
        )
        comment_id = extract_comment_id(response) or f"posted-{decision.agent.slot}-{decision.paper.id}"
        self.store.mark_comment(
            agent_slot=decision.agent.slot,
            paper_id=decision.paper.id,
            comment_id=comment_id,
            content_markdown=content,
            github_file_url=github_url,
            parent_comment_id=decision.parent_comment_id,
        )
        self.store.record_decision(decision, dry_run=False, response=response)
        self.record_action_memory(decision, response, update, role_results, retrieval_results, comments)
        return self.result_payload(decision, response, content)

    def execute_verdict(
        self, agent: PublicAgent, paper: Paper, comments: list[Comment]
    ) -> dict[str, Any] | None:
        citations = self.select_citations(comments)
        if len(citations) < self.config.min_verdict_citations:
            return None
        retrieval_results = self.retrieve(paper, agent)
        analysis_paper = self.paper_reader.enrich(paper)
        roles = [
            "claim_evidence_auditor",
            "critical_reviewer",
            "permissive_reviewer",
            "area_chair_simulator",
            "borderline_calibrator",
            "limitations_checker",
        ][: self.config.max_internal_roles]
        discussion_roles = ["discussion_analyst", "area_chair_simulator", "borderline_calibrator"]
        discussion_excerpt = discussion_excerpt_for(comments)
        role_results = [
            self.run_model_role(role, agent, analysis_paper, retrieval_results, "") for role in roles
        ]
        discussion_role_results = [
            self.run_model_role(role, agent, analysis_paper, retrieval_results, discussion_excerpt)
            for role in discussion_roles
        ]
        update = self.discussion_update(role_results, comments)
        probability = update.posterior_probability
        confidence = update.posterior_confidence
        score = confidence * (10 * probability) + (1 - confidence) * 5
        score = max(0.0, min(10.0, round(score, 2)))
        cited = " ".join(f"[[comment:{comment.id}]]" for comment in citations[:5])
        content = safe_public_markdown(
            f"Final verdict: score {score:.2f}/10.\n\n"
            f"I cite these discussion comments because they helped anchor the decision: {cited}.\n\n"
            f"Calibration: paper-only prior P={update.prior_probability:.2f}, "
            f"discussion-updated posterior P={probability:.2f}, confidence={confidence:.2f}. "
            "The score is shrunk toward 5 under uncertainty.\n\n"
            f"Discussion update: {update.rationale}\n\n"
            f"Rationale:\n{verdict_rationale([*role_results, *discussion_role_results])}"
        )
        _, github_url = self.logger.write_log(
            kind="verdict",
            agent=agent,
            paper=paper,
            content_markdown=content,
            metadata={
                "score": score,
                "accept_probability": probability,
                "confidence": confidence,
                "bayesian_update": update,
                "discussion_signals": extract_discussion_signals(comments, self.sibling_actor_ids),
                "citations": [comment.id for comment in citations],
                "role_results": role_results,
                "discussion_role_results": discussion_role_results,
            },
        )
        if self.config.dry_run:
            self.record_verdict_memory(agent, paper, score, update, role_results, comments, retrieval_results)
            return {
                "dry_run": True,
                "agent": agent.name,
                "paper_id": paper.id,
                "action": "verdict",
                "score": score,
                "github_file_url": github_url,
            }
        self.require_live_ready(agent, github_url)
        response = KoalaClient(self.config.koala_api_base, agent).post_verdict(
            paper.id, content, score, github_url
        )
        self.store.mark_verdict(
            agent_slot=agent.slot,
            paper_id=paper.id,
            score=score,
            content_markdown=content,
            github_file_url=github_url,
        )
        self.record_verdict_memory(agent, paper, score, update, role_results, comments, retrieval_results)
        return {"agent": agent.name, "paper_id": paper.id, "action": "verdict", "response": response}

    def retrieve(self, paper: Paper, agent: PublicAgent) -> list[RetrievalResult]:
        queries = retrieval_queries_for(paper, agent.public_role)
        retrieval_roles = {"literature_novelty", "meta_calibration"}
        if self.config.enable_retrieval_for_all_agents:
            retrieval_roles.add("evidence_rigor")
        if not queries or agent.public_role not in retrieval_roles:
            return []
        results = filter_target_paper_results(
            paper, self.retrieval.search_many(queries, per_query=4)
        )[:12]
        for query in queries:
            self.memory.record_retrieval(query, results)
        self.trajectory.record(
            agent=agent,
            event="retrieval",
            paper=paper,
            payload={"queries": queries, "result_count": len(results), "results": results[:5]},
        )
        return results

    def run_model_role(
        self,
        role: str,
        agent: PublicAgent,
        paper: Paper,
        retrieval_results: list[RetrievalResult],
        discussion_excerpt: str,
    ) -> RoleResult:
        return self.llm.run_role(
            role,
            paper,
            retrieval_results,
            discussion_excerpt,
            tool_executor=lambda name, arguments: self.execute_model_tool(
                agent, paper, role, name, arguments
            ),
            enable_tool_calls=self.config.enable_model_tool_calls,
            max_tool_call_rounds=self.config.max_tool_call_rounds,
        )

    def execute_model_tool(
        self,
        agent: PublicAgent,
        paper: Paper,
        role: str,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        query = " ".join(str(arguments.get("query") or "").split())
        limit = bounded_int(arguments.get("limit"), default=5, low=1, high=8)
        result: dict[str, Any]
        if name not in {"search_semantic_scholar", "search_arxiv", "search_related_work"}:
            result = {"ok": False, "error": "unknown_tool", "name": name}
        elif not safe_model_query(query, paper):
            result = {"ok": False, "error": "unsafe_or_empty_query", "query": query}
        else:
            if name == "search_semantic_scholar":
                results = self.retrieval.semantic_scholar.search(query, limit=limit)
            elif name == "search_arxiv":
                results = self.retrieval.arxiv.search(query, limit=limit)
            else:
                results = self.retrieval.search_many([query], per_query=limit)
            results = filter_target_paper_results(paper, results)[:limit]
            self.memory.record_retrieval(query, results)
            result = {
                "ok": True,
                "query": query,
                "result_count": len(results),
                "results": results,
            }
        self.trajectory.record(
            agent=agent,
            event="model_tool_call",
            paper=paper,
            payload={
                "role": role,
                "tool": name,
                "arguments": arguments,
                "result": result,
            },
        )
        if not result.get("ok"):
            self.memory.record_failure(
                "model_tool_call",
                str(result.get("error") or "tool_call_failed"),
                agent=agent,
                paper_id=paper.id,
                payload={"role": role, "tool": name, "arguments": arguments, "result": result},
            )
        return result

    def run_roles(
        self,
        decision: ActionDecision,
        comments: list[Comment],
        retrieval_results: list[RetrievalResult],
        *,
        include_discussion: bool = True,
    ) -> list[RoleResult]:
        discussion_excerpt = discussion_excerpt_for(comments) if include_discussion else ""
        return [
            self.run_model_role(role, decision.agent, decision.paper, retrieval_results, discussion_excerpt)
            for role in decision.roles
        ]

    def discussion_update(self, role_results: list[RoleResult], comments: list[Comment]):
        prior_probability, prior_confidence = aggregate_prior(role_results)
        signals = extract_discussion_signals(comments, self.sibling_actor_ids)
        if not self.config.bayesian_discussion_updates:
            return bayesian_update_from_signals(
                prior_probability,
                prior_confidence,
                [],
                max_discussion_weight=0.0,
            )
        return bayesian_update_from_signals(
            prior_probability,
            prior_confidence,
            signals,
            max_discussion_weight=self.config.max_discussion_update_weight,
        )

    def record_action_memory(
        self,
        decision: ActionDecision,
        response: dict[str, Any],
        update,
        role_results: list[RoleResult],
        retrieval_results: list[RetrievalResult],
        comments: list[Comment],
    ) -> None:
        signals = extract_discussion_signals(comments, self.sibling_actor_ids)
        payload = {
            "response": response,
            "bayesian_update": update,
            "role_results": role_results,
            "retrieval_results": retrieval_results,
            "discussion_signal_count": len(signals),
        }
        self.memory.record_comment_action(decision.agent, decision.paper.id, decision.action, decision.ev, payload)
        self.memory.record_external_signals(signals)
        self.trajectory.record(
            agent=decision.agent,
            event=f"action_{decision.action}",
            paper=decision.paper,
            payload={"ev": decision.ev, "reason": decision.reason, **payload},
        )

    def record_verdict_memory(
        self,
        agent: PublicAgent,
        paper: Paper,
        score: float,
        update,
        role_results: list[RoleResult],
        comments: list[Comment],
        retrieval_results: list[RetrievalResult],
    ) -> None:
        signals = extract_discussion_signals(comments, self.sibling_actor_ids)
        payload = {
            "retrieval_results": retrieval_results,
            "discussion_signal_count": len(signals),
            "bayesian_update": update,
        }
        self.memory.record_verdict(agent, paper.id, score, update, role_results, paper.domains, payload)
        self.memory.record_external_signals(signals)
        self.trajectory.record(
            agent=agent,
            event="verdict",
            paper=paper,
            payload={"score": score, "role_results": role_results, **payload},
        )

    def select_citations(self, comments: list[Comment]) -> list[Comment]:
        external = [
            comment
            for comment in comments
            if comment.id and comment.author_id and comment.author_id not in self.sibling_actor_ids
        ]
        by_author: dict[str, Comment] = {}
        for comment in sorted(external, key=lambda item: comment_quality_for_verdict(item), reverse=True):
            by_author.setdefault(comment.author_id, comment)
        return list(by_author.values())[:5]

    def require_live_ready(self, agent: PublicAgent, github_url: str) -> None:
        if not agent.api_key:
            raise RuntimeError(f"{agent.name} has no Koala API key configured")
        if "github.com" not in github_url or "REPLACE_ME" in github_url:
            raise RuntimeError("GITHUB_REPO_URL must be configured before live posting")

    def result_payload(
        self, decision: ActionDecision, response: dict[str, Any], content: str
    ) -> dict[str, Any]:
        return {
            "agent": decision.agent.name,
            "paper_id": decision.paper.id,
            "paper_title": decision.paper.title,
            "action": decision.action,
            "ev": round(decision.ev, 3),
            "reason": decision.reason,
            "response": response,
            "content_preview": content[:500],
        }


def discussion_excerpt_for(comments: list[Comment], *, limit: int = 6) -> str:
    chosen = sorted(comments, key=comment_quality_for_verdict, reverse=True)[:limit]
    return "\n\n".join(
        f"Comment {comment.id} by {comment.author_id}:\n{comment.content_markdown[:1200]}"
        for comment in chosen
    )


def comment_quality_for_verdict(comment: Comment) -> float:
    text = comment.content_markdown.lower()
    score = min(1.0, len(comment.content_markdown) / 1400)
    score += 0.2 if any(term in text for term in ["section", "table", "figure", "baseline"]) else 0
    score += 0.2 if any(term in text for term in ["evidence", "claim", "novelty", "ablation"]) else 0
    return score


def verdict_rationale(role_results: list[RoleResult]) -> str:
    lines: list[str] = []
    for result in role_results:
        lines.append(f"- {result.role_name}: P={result.accept_probability:.2f}, confidence={result.confidence:.2f}.")
        for fatal in result.fatal_flaws[:1]:
            lines.append(f"  Fatal-flaw candidate: {fatal}")
        for weakness in result.weaknesses[:2]:
            lines.append(f"  Concern: {weakness}")
        for strength in result.strengths[:1]:
            lines.append(f"  Strength: {strength}")
    return "\n".join(lines)


def extract_comment_id(response: dict[str, Any]) -> str:
    direct = response.get("id") or response.get("comment_id")
    if direct:
        return str(direct)
    for key in ("comment", "data", "result"):
        nested = response.get(key)
        if isinstance(nested, dict):
            nested_id = nested.get("id") or nested.get("comment_id")
            if nested_id:
                return str(nested_id)
    return ""


def safe_public_markdown(content: str, *, max_chars: int = 4500) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Keep comments substantial but not sprawling; detailed reasoning is preserved in logs.
    if len(content) > max_chars:
        content = content[: max_chars - 120].rstrip() + "\n\n[Truncated for readability; full reasoning is in the linked transparency log.]"
    banned_fragments = ["idiot", "stupid", "nonsense"]
    for fragment in banned_fragments:
        content = content.replace(fragment, "unsupported")
    return content


def append_bayesian_note(content: str, update, *, include_when_no_signals: bool = False) -> str:
    if update.signal_count == 0 and not include_when_no_signals:
        return content
    if update.signal_count == 0:
        note = (
            "\n\nCalibration note: this is currently a paper-only prior; I did not find useful "
            "external discussion signals to update on yet."
        )
    else:
        note = (
            "\n\nCalibration note: my paper-only prior is "
            f"P(accept)={update.prior_probability:.2f}. After weighting external comments by "
            "evidence quality, confidence, independence, and relevance, the posterior is "
            f"P(accept)={update.posterior_probability:.2f}. "
            f"{update.rationale}"
        )
    return safe_public_markdown(content + note)


def safe_model_query(query: str, paper: Paper) -> bool:
    if not query:
        return False
    lowered = query.lower()
    if any(term in lowered for term in ["openreview", "anonymous authors", "submitted to icml"]):
        return False
    title_key = normalize_query_key(paper.title)
    query_key = normalize_query_key(query)
    if title_key and len(title_key) > 24 and title_key in query_key:
        return False
    if re.search(r"\b(author|authors|by|from)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b", query):
        return False
    tokens = query.split()
    if len(tokens) <= 3 and re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", query):
        return False
    return True


def normalize_query_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def filter_target_paper_results(paper: Paper, results: list[RetrievalResult]) -> list[RetrievalResult]:
    target_key = normalize_query_key(paper.title)
    if not target_key:
        return results
    filtered: list[RetrievalResult] = []
    for result in results:
        result_key = normalize_query_key(result.title)
        if result_key and len(target_key) > 24 and (result_key == target_key or target_key in result_key):
            continue
        filtered.append(result)
    return filtered


def bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def profile_description(public_role: str) -> str:
    descriptions = {
        "evidence_rigor": "Dynamic Gemini reviewer focused on claim-evidence alignment, experimental rigor, reproducibility, benchmarks, and code-method consistency.",
        "literature_novelty": "Dynamic Gemini reviewer focused on related work, novelty, citation factuality, prior-art positioning, and claim calibration.",
        "meta_calibration": "Dynamic Gemini reviewer focused on discussion synthesis, disagreement analysis, area-chair-style calibration, and final accept-probability estimates.",
    }
    return descriptions.get(public_role, "Dynamic Gemini peer-review agent with transparent logs and EV-based participation policy.")
