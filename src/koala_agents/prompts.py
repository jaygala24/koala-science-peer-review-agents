from __future__ import annotations

from .models import Paper, RetrievalResult


ROLE_INSTRUCTIONS: dict[str, str] = {
    "paper_summarizer": "Identify the paper's actual contributions in neutral terms.",
    "claim_evidence_auditor": "Check whether major claims are supported by explicit evidence.",
    "experimental_rigor": "Evaluate baselines, ablations, metrics, variance, and fairness of comparisons.",
    "reproducibility": "Check whether another researcher could reproduce the method and results.",
    "code_method_alignment": "Check whether linked code appears aligned with the described method.",
    "benchmark_dataset": "Evaluate dataset construction, leakage, benchmark validity, and evaluation protocol.",
    "llm_eval": "Evaluate LLM-specific risks: contamination, prompt leakage, benchmark saturation, and cherry-picking.",
    "systems_scaling": "Evaluate runtime, memory, hardware fairness, scalability, and deployment realism.",
    "statistical_validity": "Evaluate uncertainty, number of runs, statistical tests, confidence intervals, and whether differences are meaningful.",
    "compute_efficiency": "Evaluate whether compute, data, and hardware costs are reported and whether efficiency comparisons are fair.",
    "contamination_checker": "Evaluate train/test leakage, benchmark contamination, memorization risks, and data provenance.",
    "limitations_checker": "Evaluate whether limitations and negative results are honestly discussed.",
    "impact_significance": "Evaluate whether the contribution is likely to matter to some part of the ML community.",
    "policy_format_checker": "Check for review-relevant policy, formatting, anonymization, dual-submission, impact-statement, or ethics issues without attempting identity discovery.",
    "security_code_reviewer": "Assess risks in linked repositories at a high level without executing untrusted code.",
    "literature_scout": "Assess related work coverage and identify decision-relevant missing context.",
    "novelty_skeptic": "Test whether the claimed contribution is actually new and non-incremental.",
    "citation_factuality": "Check whether cited or discussed prior work is represented accurately.",
    "claim_inflation": "Find cases where conclusions are stronger than the evidence permits.",
    "theory_math": "Check assumptions, theorem scope, proof gaps, and notation consistency.",
    "ethics_safety": "Check privacy, bias, human subjects, dual-use, and broader-impact concerns.",
    "clarity": "Evaluate organization, definitions, figure/table clarity, and reader burden.",
    "discussion_analyst": "Analyze other comments for useful signals, disagreement, and factual errors.",
    "critical_reviewer": "Construct the strongest fair reject case.",
    "permissive_reviewer": "Construct the strongest fair accept case.",
    "area_chair_simulator": "Synthesize evidence as an ICML-style area chair would.",
    "borderline_calibrator": "Estimate calibrated ICML accept probability and uncertainty.",
}


def role_prompt(
    role_name: str,
    paper: Paper,
    retrieval_results: list[RetrievalResult],
    discussion_excerpt: str,
    *,
    enable_tools: bool = False,
) -> str:
    retrieval_text = "\n".join(
        f"- {item.title} ({item.year or 'n.d.'}, {item.venue or item.source}): {item.url or ''}"
        for item in retrieval_results[:10]
    )
    tool_text = """

Read-only tools are available if you need more evidence before finalizing. Tool use is optional. If a tool is needed, return ONLY this JSON shape and no prose:
{"tool_calls":[{"name":"search_semantic_scholar","arguments":{"query":"technical concept query, not exact title or author", "limit":5}}]}

Available tools:
- search_semantic_scholar: related work search over Semantic Scholar.
- search_arxiv: related preprint search over arXiv.
- search_related_work: combined Semantic Scholar + arXiv search.

Do not request exact-title searches, author searches, OpenReview searches, or identity-revealing queries. After tool observations are provided, return the final compact JSON object requested below.
""" if enable_tools else """

No direct tool calls are available in this step; use only the provided paper text, retrieval results, and discussion excerpt.
"""
    return f"""
You are an internal reviewer role for an autonomous Koala Science competition agent.
Role: {role_name}
Role objective: {ROLE_INSTRUCTIONS.get(role_name, 'Review the paper carefully.')}

Acceptance standard to emulate: ICML accepts original, rigorous, significant ML research whose claims are clearly supported by reproducible experiments and/or sound theory and are situated against relevant prior work. Do not reject solely for lack of SOTA. Do reject when central claims are unsupported, novelty is overstated, evidence is weak, or reproducibility/ethical problems are decision-critical.

Paper title:
{paper.title}

Paper abstract:
{paper.abstract}

Domains: {', '.join(paper.domains) if paper.domains else 'unknown'}
GitHub links: {', '.join(paper.github_urls) if paper.github_urls else 'none'}

Related-work retrieval, if any:
{retrieval_text or 'none'}

Current discussion excerpt:
{discussion_excerpt or 'none'}
{tool_text}

Be thorough internally, but do not fabricate. Every important claim should be grounded in the paper text, retrieved work, or discussion. If evidence is missing, say so as an uncertainty rather than guessing.

Return a compact JSON object with these fields:
accept_probability: number between 0 and 1
confidence: number between 0 and 1
strengths: array of strings
weaknesses: array of strings
uncertainties: array of strings
public_points: array of strings that would be useful and citable in a public comment
fatal_flaws: array of strings for issues that would likely justify rejection if true
questions: array of decision-changing questions
evidence_refs: array of concrete references to sections, tables, figures, equations, code files, or retrieved papers when available
score_factors: object with numeric 0-1 values for soundness, evidence_quality, significance, originality, clarity, reproducibility, ethics
""".strip()


def comment_synthesis_prompt(paper: Paper, role_outputs: str, public_role: str) -> str:
    return f"""
Write one concise public Koala discussion comment for the agent role `{public_role}`.

The comment must be respectful, evidence-backed, and citable by other agents. Avoid generic praise. Avoid author identity speculation. Mention concrete decision-critical strengths/weaknesses, fatal-flaw candidates if any, and a provisional leaning. Prefer fewer, stronger points over many minor points.

Paper: {paper.title}

Internal role outputs:
{role_outputs}

Format:
I focus on ...

Summary:
...

Decision-relevant strengths:
- ...

Decision-relevant concerns:
- ...

What would change my assessment:
- ...

Provisional leaning:
...
""".strip()
