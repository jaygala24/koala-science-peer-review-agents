from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .models import ActionDecision, Comment, OpportunityFeatures, Paper, PublicAgent
from .roles import classify_paper, select_roles
from .storage import AgentStore


class DynamicStrategy:
    def __init__(
        self,
        store: AgentStore,
        sibling_actor_ids: set[str] | None = None,
        *,
        exhaustive: bool = True,
        max_roles: int = 12,
        enable_dynamic_pacing: bool = True,
        max_comment_karma_per_agent_per_day: float = 8.0,
        comment_karma_budget_per_agent: float = 55.0,
        pacing_lookback_hours: float = 24.0,
    ) -> None:
        self.store = store
        self.sibling_actor_ids = sibling_actor_ids or set()
        self.exhaustive = exhaustive
        self.max_roles = max_roles
        self.enable_dynamic_pacing = enable_dynamic_pacing
        self.max_comment_karma_per_agent_per_day = max_comment_karma_per_agent_per_day
        self.comment_karma_budget_per_agent = comment_karma_budget_per_agent
        self.pacing_lookback_hours = pacing_lookback_hours

    def decide(self, paper: Paper, comments: list[Comment], agent: PublicAgent) -> ActionDecision:
        roles = select_roles(
            agent.public_role,
            paper,
            comments,
            exhaustive=self.exhaustive,
            max_roles=self.max_roles,
        )
        features = self.features(paper, comments, agent)
        already_commented = self.store.agent_has_commented(agent.slot, paper.id)

        if paper.status != "in_review":
            return ActionDecision(
                action="observe",
                agent=agent,
                paper=paper,
                ev=0.0,
                features=features,
                roles=roles,
                reason=f"paper status is {paper.status}; commenting is closed",
            )

        if already_commented:
            parent = self.best_reply_target(comments)
            ev = self.reply_ev(features, parent is not None)
            if not self.can_spend(agent, cost=0.1):
                return ActionDecision(
                    action="observe",
                    agent=agent,
                    paper=paper,
                    ev=ev,
                    features=features,
                    roles=roles,
                    reason="agent is at pacing or reserve limit for paid replies",
                )
            if parent and ev >= self.dynamic_threshold(agent, cost=0.1):
                return ActionDecision(
                    action="reply",
                    agent=agent,
                    paper=paper,
                    ev=ev,
                    features=features,
                    roles=roles,
                    reason="cheap reply has positive value for discussion quality, citation ancestry, or prediction calibration",
                    parent_comment_id=parent.id,
                )
            return ActionDecision(
                action="observe",
                agent=agent,
                paper=paper,
                ev=ev,
                features=features,
                roles=roles,
                reason="agent already commented; no high-value reply target found",
            )

        if not self.can_spend(agent, cost=1.0):
            return ActionDecision(
                action="skip",
                agent=agent,
                paper=paper,
                ev=-1.0,
                features=features,
                roles=roles,
                reason="agent karma is at or below reserve",
            )

        ev = self.first_comment_ev(features)
        if ev >= self.dynamic_threshold(agent, cost=1.0):
            return ActionDecision(
                action="first_comment",
                agent=agent,
                paper=paper,
                ev=ev,
                features=features,
                roles=roles,
                reason="first comment appears to buy a positive-EV verdict option and/or citation opportunity",
            )
        if ev > 0:
            action = "observe"
            reason = "positive but below threshold; preserve optionality"
        else:
            action = "skip"
            reason = "expected value does not justify spending 1 karma"
        return ActionDecision(
            action=action,
            agent=agent,
            paper=paper,
            ev=ev,
            features=features,
            roles=roles,
            reason=reason,
        )

    def features(self, paper: Paper, comments: list[Comment], agent: PublicAgent) -> OpportunityFeatures:
        participant_count = len({comment.author_id for comment in comments if comment.author_id})
        external_authors = self.external_author_ids(comments)
        external_count = len(external_authors)
        hours_left = self.hours_until_comment_close(paper)
        touched_slots = self.store.touched_agent_slots(paper.id)

        if external_count >= 3:
            verdict_feasibility = 1.0
        elif external_count == 2 and hours_left > 12:
            verdict_feasibility = 0.75
        elif external_count == 1 and hours_left > 18:
            verdict_feasibility = 0.45
        elif hours_left > 30:
            verdict_feasibility = 0.25
        else:
            verdict_feasibility = 0.05

        domain_fit = self.domain_fit(paper, agent.public_role)
        under_reviewed_bonus = max(0.0, min(1.0, (10 - participant_count) / 10))
        discussion_quality = self.discussion_quality(comments)
        citation_probability = min(
            1.0,
            0.25 + 0.35 * discussion_quality + 0.25 * under_reviewed_bonus + 0.15 * domain_fit,
        )
        karma_recovery_potential = min(
            1.0, 0.4 * citation_probability + 0.3 * under_reviewed_bonus + 0.3 * verdict_feasibility
        )
        sibling_overlap_penalty = 0.0 if not touched_slots - {agent.slot} else 0.7
        time_risk = self.time_risk(hours_left, external_count)
        crowding_penalty = max(0.0, min(1.0, (participant_count - 8) / 8))
        redundancy_penalty = self.redundancy_penalty(comments, agent.public_role)
        prediction_edge = min(1.0, 0.45 * domain_fit + 0.25 * under_reviewed_bonus + 0.3)

        return OpportunityFeatures(
            verdict_feasibility=verdict_feasibility,
            domain_fit=domain_fit,
            prediction_edge=prediction_edge,
            under_reviewed_bonus=under_reviewed_bonus,
            discussion_quality=discussion_quality,
            citation_probability=citation_probability,
            karma_recovery_potential=karma_recovery_potential,
            sibling_overlap_penalty=sibling_overlap_penalty,
            time_risk=time_risk,
            crowding_penalty=crowding_penalty,
            redundancy_penalty=redundancy_penalty,
        )

    def first_comment_ev(self, features: OpportunityFeatures) -> float:
        gross = (
            2.0 * features.verdict_feasibility
            + 1.7 * features.domain_fit
            + 1.5 * features.prediction_edge
            + 1.2 * features.under_reviewed_bonus
            + 1.0 * features.discussion_quality
            + 1.0 * features.citation_probability
            + 0.8 * features.karma_recovery_potential
        )
        penalties = (
            1.5 * features.sibling_overlap_penalty
            + 1.2 * features.time_risk
            + 1.0 * features.crowding_penalty
            + 1.0 * features.redundancy_penalty
        )
        return gross - penalties - 1.0

    def reply_ev(self, features: OpportunityFeatures, has_parent: bool) -> float:
        if not has_parent:
            return -0.1
        gross = (
            0.8 * features.discussion_quality
            + 0.8 * features.citation_probability
            + 0.6 * features.karma_recovery_potential
            + 0.4 * features.prediction_edge
        )
        penalties = 0.5 * features.redundancy_penalty + 0.4 * features.crowding_penalty
        return gross - penalties - 0.1

    def dynamic_threshold(self, agent: PublicAgent, *, cost: float) -> float:
        if agent.karma is None:
            reserve_pressure = 0.0
        else:
            cushion = agent.karma - agent.min_karma_reserve
            reserve_pressure = 0.4 if cushion < 5 else 0.2 if cushion < 15 else 0.0
        return (0.25 if cost >= 1.0 else 0.05) + reserve_pressure + self.pacing_pressure(agent, cost=cost)

    def can_spend(self, agent: PublicAgent, *, cost: float) -> bool:
        if agent.karma is None:
            reserve_ok = True
        else:
            reserve_ok = agent.karma - cost >= agent.min_karma_reserve
        if not reserve_ok:
            return False
        if not self.enable_dynamic_pacing or self.comment_karma_budget_per_agent <= 0:
            return True
        spent = self.store.paid_comment_karma_spent(agent.slot, live_only=True)
        return spent + cost <= self.comment_karma_budget_per_agent

    def pacing_pressure(self, agent: PublicAgent, *, cost: float) -> float:
        if not self.enable_dynamic_pacing or self.max_comment_karma_per_agent_per_day <= 0:
            return 0.0
        since = datetime.now(tz=UTC) - timedelta(hours=max(1.0, self.pacing_lookback_hours))
        recent_spend = self.store.paid_comment_karma_spent(agent.slot, since=since, live_only=True)
        daily_budget = self.max_comment_karma_per_agent_per_day * max(1.0, self.pacing_lookback_hours) / 24.0
        pressure_ratio = (recent_spend + cost) / max(0.1, daily_budget)
        if pressure_ratio <= 0.5:
            return -0.05 if cost >= 1.0 else -0.02
        if pressure_ratio <= 1.0:
            return 0.25 * ((pressure_ratio - 0.5) / 0.5)
        return 0.75 + min(1.5, pressure_ratio - 1.0)

    def external_author_ids(self, comments: list[Comment]) -> set[str]:
        return {
            comment.author_id
            for comment in comments
            if comment.author_id and comment.author_id not in self.sibling_actor_ids
        }

    def best_reply_target(self, comments: list[Comment]) -> Comment | None:
        candidates = [
            comment
            for comment in comments
            if comment.author_id not in self.sibling_actor_ids and len(comment.content_markdown) >= 250
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda comment: self.comment_quality(comment))

    def comment_quality(self, comment: Comment) -> float:
        text = comment.content_markdown.lower()
        score = min(1.0, len(comment.content_markdown) / 1200)
        score += 0.15 if any(token in text for token in ["section", "table", "figure", "baseline"]) else 0
        score += 0.15 if any(token in text for token in ["claim", "evidence", "ablation", "novelty"]) else 0
        return min(1.0, score)

    def hours_until_comment_close(self, paper: Paper) -> float:
        if paper.created_at is None:
            return 24.0
        age_hours = (datetime.now(tz=UTC) - paper.created_at).total_seconds() / 3600
        return max(0.0, 48.0 - age_hours)

    def time_risk(self, hours_left: float, external_count: int) -> float:
        if external_count >= 3:
            return 0.0 if hours_left > 0 else 0.2
        if hours_left < 4:
            return 1.0
        if hours_left < 12:
            return 0.7
        if hours_left < 24 and external_count == 0:
            return 0.5
        return 0.15

    def discussion_quality(self, comments: list[Comment]) -> float:
        if not comments:
            return 0.0
        qualities = [self.comment_quality(comment) for comment in comments]
        return sum(qualities) / len(qualities)

    def domain_fit(self, paper: Paper, public_role: str) -> float:
        labels = classify_paper(paper)
        if public_role == "evidence_rigor":
            return 1.0 if labels & {"empirical", "llm", "systems", "code"} else 0.55
        if public_role == "literature_novelty":
            return 1.0 if labels & {"method", "llm", "theory"} else 0.65
        if public_role == "meta_calibration":
            return 0.85
        return 0.6

    def redundancy_penalty(self, comments: list[Comment], public_role: str) -> float:
        if not comments:
            return 0.0
        text = "\n".join(comment.content_markdown.lower() for comment in comments)
        role_terms = {
            "evidence_rigor": ["baseline", "ablation", "reproducib", "experiment"],
            "literature_novelty": ["novelty", "related work", "prior work", "citation"],
            "meta_calibration": ["borderline", "synthesis", "overall", "leaning"],
        }.get(public_role, [])
        hits = sum(1 for term in role_terms if term in text)
        return min(0.8, hits * 0.18)
