from __future__ import annotations

import math
import re

from .models import BayesianUpdate, Comment, DiscussionSignal, RoleResult


def aggregate_prior(role_results: list[RoleResult]) -> tuple[float, float]:
    if not role_results:
        return 0.5, 0.35
    total_weight = sum(max(0.05, result.confidence) for result in role_results)
    probability = sum(result.accept_probability * max(0.05, result.confidence) for result in role_results)
    probability /= total_weight
    confidence = sum(result.confidence for result in role_results) / len(role_results)
    fatal_count = sum(1 for result in role_results if result.fatal_flaws)
    if fatal_count:
        probability = max(0.02, probability - min(0.25, 0.08 * fatal_count))
        confidence = min(1.0, confidence + min(0.12, 0.03 * fatal_count))
    return clamp(probability), clamp(confidence)


def extract_discussion_signals(
    comments: list[Comment], sibling_actor_ids: set[str]
) -> list[DiscussionSignal]:
    signals: list[DiscussionSignal] = []
    seen_authors: set[str] = set()
    for comment in comments:
        if not comment.id or not comment.author_id or comment.author_id in sibling_actor_ids:
            continue
        signal = heuristic_discussion_signal(comment, author_seen=comment.author_id in seen_authors)
        seen_authors.add(comment.author_id)
        if signal.useful:
            signals.append(signal)
    return signals


def bayesian_update_from_signals(
    prior_probability: float,
    prior_confidence: float,
    signals: list[DiscussionSignal],
    *,
    max_discussion_weight: float = 0.35,
) -> BayesianUpdate:
    useful_signals = [signal for signal in signals if signal.useful]
    if not useful_signals:
        return BayesianUpdate(
            prior_probability=prior_probability,
            prior_confidence=prior_confidence,
            posterior_probability=prior_probability,
            posterior_confidence=prior_confidence,
            discussion_weight=0.0,
            signal_count=0,
            weighted_signal_probability=None,
            rationale="No useful external discussion signals; posterior equals prior.",
        )

    weights = [signal_weight(signal) for signal in useful_signals]
    total = sum(weights)
    if total <= 0:
        return BayesianUpdate(
            prior_probability=prior_probability,
            prior_confidence=prior_confidence,
            posterior_probability=prior_probability,
            posterior_confidence=prior_confidence,
            discussion_weight=0.0,
            signal_count=len(useful_signals),
            weighted_signal_probability=None,
            rationale="External signals had zero effective reliability; posterior equals prior.",
        )
    signal_probability = sum(
        signal.stance_probability * weight for signal, weight in zip(useful_signals, weights, strict=True)
    ) / total

    disagreement = weighted_disagreement(useful_signals, weights, signal_probability)
    evidence_mass = min(1.0, total / 4.0)
    raw_discussion_weight = max_discussion_weight * evidence_mass * (1.0 - 0.45 * disagreement)
    discussion_weight = clamp(raw_discussion_weight, 0.0, max_discussion_weight)

    prior_logit = logit(prior_probability)
    signal_logit = logit(signal_probability)
    posterior_probability = sigmoid((1 - discussion_weight) * prior_logit + discussion_weight * signal_logit)

    avg_signal_confidence = sum(signal.confidence * weight for signal, weight in zip(useful_signals, weights, strict=True)) / total
    posterior_confidence = clamp(
        prior_confidence * (1 - 0.25 * disagreement)
        + discussion_weight * avg_signal_confidence * (1 - disagreement)
    )

    rationale = (
        f"Updated prior P={prior_probability:.2f} using {len(useful_signals)} external signals. "
        f"Weighted signal P={signal_probability:.2f}, discussion weight={discussion_weight:.2f}, "
        f"disagreement={disagreement:.2f}."
    )
    return BayesianUpdate(
        prior_probability=clamp(prior_probability),
        prior_confidence=clamp(prior_confidence),
        posterior_probability=clamp(posterior_probability),
        posterior_confidence=posterior_confidence,
        discussion_weight=discussion_weight,
        signal_count=len(useful_signals),
        weighted_signal_probability=clamp(signal_probability),
        rationale=rationale,
    )


def heuristic_discussion_signal(comment: Comment, *, author_seen: bool) -> DiscussionSignal:
    text = comment.content_markdown.lower()
    stance = 0.5
    stance += 0.18 if any(term in text for term in ["accept", "strong accept", "weak accept"]) else 0
    stance -= 0.18 if any(term in text for term in ["reject", "weak reject", "clear reject"]) else 0
    stance += 0.08 if any(term in text for term in ["strong contribution", "technically sound", "convincing"] ) else 0
    stance -= 0.08 if any(term in text for term in ["unsupported", "not convincing", "missing baseline", "fatal", "flaw"] ) else 0

    confidence = 0.25
    confidence += min(0.3, len(comment.content_markdown) / 3000)
    confidence += 0.12 if has_evidence_anchor(text) else 0
    confidence += 0.08 if any(term in text for term in ["because", "therefore", "specifically"]) else 0
    confidence -= 0.12 if any(term in text for term in ["maybe", "unclear", "not sure", "i guess"] ) else 0

    evidence_quality = 0.25
    evidence_quality += 0.25 if has_evidence_anchor(text) else 0
    evidence_quality += 0.15 if any(term in text for term in ["baseline", "ablation", "theorem", "table", "figure", "section"] ) else 0
    evidence_quality += 0.1 if any(term in text for term in ["related work", "prior work", "citation"] ) else 0
    evidence_quality -= 0.15 if len(comment.content_markdown) < 160 else 0

    relevance = 0.35
    relevance += 0.25 if any(term in text for term in ["claim", "evidence", "experiment", "proof", "novelty", "reproduc"] ) else 0
    relevance += 0.15 if any(term in text for term in ["accept", "reject", "score", "verdict", "decision"] ) else 0

    independence = 0.55 if not author_seen else 0.25
    if "building on" in text or "as noted" in text:
        independence -= 0.15

    useful = evidence_quality >= 0.25 and relevance >= 0.35
    return DiscussionSignal(
        comment_id=comment.id,
        author_id=comment.author_id,
        stance_probability=clamp(stance),
        confidence=clamp(confidence),
        evidence_quality=clamp(evidence_quality),
        independence=clamp(independence),
        relevance=clamp(relevance),
        summary=summarize_comment(comment.content_markdown),
        useful=useful,
    )


def signal_weight(signal: DiscussionSignal) -> float:
    return (
        0.35 * signal.confidence
        + 0.30 * signal.evidence_quality
        + 0.20 * signal.independence
        + 0.15 * signal.relevance
    )


def weighted_disagreement(
    signals: list[DiscussionSignal], weights: list[float], center: float
) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    variance = sum(weight * (signal.stance_probability - center) ** 2 for signal, weight in zip(signals, weights, strict=True)) / total
    return clamp(math.sqrt(variance) * 2.0)


def has_evidence_anchor(text: str) -> bool:
    return bool(
        re.search(r"\b(section|sec\.|table|figure|fig\.|appendix|theorem|equation|eq\.)\b", text)
        or re.search(r"\b\d+(\.\d+)?\s*(%|points?|runs?|seeds?)\b", text)
    )


def summarize_comment(content: str) -> str:
    content = " ".join(content.split())
    return content[:240] + ("..." if len(content) > 240 else "")


def logit(probability: float) -> float:
    probability = clamp(probability, 0.01, 0.99)
    return math.log(probability / (1 - probability))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
