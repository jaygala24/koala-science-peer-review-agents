from __future__ import annotations

from .models import Comment, Paper


def classify_paper(paper: Paper) -> set[str]:
    text = f"{paper.title}\n{paper.abstract}\n{' '.join(paper.domains)}".lower()
    labels: set[str] = set()
    if any(term in text for term in ["theorem", "proof", "bound", "convergence", "regret"]):
        labels.add("theory")
    if any(term in text for term in ["benchmark", "dataset", "evaluation", "experiment"]):
        labels.add("empirical")
    if any(term in text for term in ["dataset", "benchmark", "corpus", "data collection", "annotations"]):
        labels.add("dataset")
    if any(term in text for term in ["large language", "llm", "language model", "prompt"]):
        labels.add("llm")
    if any(term in text for term in ["system", "latency", "throughput", "memory", "distributed"]):
        labels.add("systems")
    if any(term in text for term in ["novel", "new", "first", "framework", "method"]):
        labels.add("method")
    if paper.github_urls:
        labels.add("code")
    if not labels:
        labels.add("general")
    return labels


def select_roles(
    public_role: str,
    paper: Paper,
    comments: list[Comment],
    *,
    exhaustive: bool = True,
    max_roles: int = 12,
) -> list[str]:
    labels = classify_paper(paper)
    roles: list[str] = ["paper_summarizer", "claim_evidence_auditor"]

    if public_role == "evidence_rigor":
        if "theory" in labels and "empirical" not in labels:
            roles += ["theory_math", "clarity"]
        else:
            roles += [
                "experimental_rigor",
                "benchmark_dataset",
                "statistical_validity",
                "reproducibility",
                "limitations_checker",
            ]
        if "code" in labels:
            roles += ["code_method_alignment", "security_code_reviewer"]
        if "llm" in labels:
            roles += ["llm_eval", "contamination_checker"]
        if "systems" in labels:
            roles += ["systems_scaling", "compute_efficiency"]

    elif public_role == "literature_novelty":
        roles += [
            "literature_scout",
            "novelty_skeptic",
            "citation_factuality",
            "claim_inflation",
            "impact_significance",
            "limitations_checker",
        ]
        if "llm" in labels:
            roles.append("llm_eval")
        if "dataset" in labels:
            roles += ["ethics_safety", "policy_format_checker"]

    elif public_role == "meta_calibration":
        roles += ["discussion_analyst", "critical_reviewer", "permissive_reviewer"]
        roles += ["area_chair_simulator", "borderline_calibrator", "impact_significance"]
        if len(comments) < 3:
            roles.append("claim_evidence_auditor")

    else:
        roles += ["experimental_rigor", "literature_scout", "borderline_calibrator"]

    roles = dedupe(roles)
    if exhaustive:
        for role in [
            "clarity",
            "limitations_checker",
            "policy_format_checker",
            "ethics_safety",
        ]:
            if role not in roles:
                roles.append(role)
    return roles[:max_roles]


def retrieval_queries_for(paper: Paper, public_role: str) -> list[str]:
    # Use abstract/domains rather than exact title to reduce identity-leakage risk.
    text = f"{paper.abstract} {' '.join(paper.domains)}"
    tokens = [token.strip(".,:;()[]{}") for token in text.split()]
    keywords = [
        token
        for token in tokens
        if len(token) > 5 and token.lower() not in COMMON_WORDS and not token[0].isupper()
    ][:10]
    base = " ".join(keywords[:6])
    queries: list[str] = []
    if base:
        queries.append(base)
    if public_role == "literature_novelty" and keywords:
        queries.append(" ".join([*keywords[:4], "related work baseline"]))
    if "benchmark" in text.lower() or "dataset" in text.lower():
        queries.append(" ".join([*keywords[:4], "benchmark dataset evaluation"] ))
    return [query for query in queries if query.strip()]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


COMMON_WORDS = {
    "abstract",
    "method",
    "model",
    "models",
    "paper",
    "using",
    "based",
    "learning",
    "approach",
    "results",
    "propose",
    "proposed",
    "show",
    "shows",
    "improve",
    "improves",
    "performance",
    "through",
    "across",
    "different",
    "abstract",
    "introduction",
}
