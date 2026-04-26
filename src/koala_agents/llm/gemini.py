from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..models import Paper, RetrievalResult, RoleResult
from ..prompts import comment_synthesis_prompt, role_prompt


@dataclass
class GeminiRunner:
    api_key: str
    model: str
    dry_run: bool = True

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False
        return True

    def generate_text(self, prompt: str) -> str:
        if not self.available():
            return ""
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model, contents=prompt)
        return getattr(response, "text", "") or ""

    def run_role(
        self,
        role_name: str,
        paper: Paper,
        retrieval_results: list[RetrievalResult],
        discussion_excerpt: str,
        *,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        enable_tool_calls: bool = False,
        max_tool_call_rounds: int = 2,
    ) -> RoleResult:
        prompt = role_prompt(
            role_name,
            paper,
            retrieval_results,
            discussion_excerpt,
            enable_tools=enable_tool_calls and tool_executor is not None,
        )
        text = self.generate_with_optional_tools(
            prompt,
            tool_executor=tool_executor,
            enable_tool_calls=enable_tool_calls,
            max_tool_call_rounds=max_tool_call_rounds,
        )
        if not text:
            return heuristic_role_result(role_name, paper, discussion_excerpt)
        parsed = parse_json_object(text)
        if not parsed:
            return heuristic_role_result(role_name, paper, discussion_excerpt, raw_text=text)
        return RoleResult(
            role_name=role_name,
            accept_probability=clamp(float(parsed.get("accept_probability", 0.5))),
            confidence=clamp(float(parsed.get("confidence", 0.45))),
            strengths=[str(item) for item in parsed.get("strengths", [])][:5],
            weaknesses=[str(item) for item in parsed.get("weaknesses", [])][:5],
            uncertainties=[str(item) for item in parsed.get("uncertainties", [])][:5],
            public_points=[str(item) for item in parsed.get("public_points", [])][:5],
            fatal_flaws=[str(item) for item in parsed.get("fatal_flaws", [])][:5],
            questions=[str(item) for item in parsed.get("questions", [])][:5],
            evidence_refs=[str(item) for item in parsed.get("evidence_refs", [])][:8],
            score_factors=parse_score_factors(parsed.get("score_factors", {})),
            raw_text=text,
        )

    def generate_with_optional_tools(
        self,
        prompt: str,
        *,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None,
        enable_tool_calls: bool,
        max_tool_call_rounds: int,
    ) -> str:
        text = self.generate_text(prompt)
        if not enable_tool_calls or tool_executor is None or not text:
            return text
        transcript = prompt
        final_text = text
        for _ in range(max(0, max_tool_call_rounds)):
            parsed = parse_json_object(final_text)
            tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
            if not isinstance(tool_calls, list) or not tool_calls:
                return final_text
            observations = []
            for call in tool_calls[:4]:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                observations.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": tool_executor(name, arguments),
                    }
                )
            transcript = (
                f"{transcript}\n\nModel tool request:\n{final_text}\n\n"
                f"Tool observations:\n{json.dumps(to_jsonable(observations), indent=2, sort_keys=True)}\n\n"
                "Now return the final compact JSON object with accept_probability, confidence, strengths, weaknesses, uncertainties, public_points, fatal_flaws, questions, evidence_refs, and score_factors. Do not request more tools unless absolutely necessary."
            )
            final_text = self.generate_text(transcript)
            if not final_text:
                return text
        return final_text

    def synthesize_comment(self, paper: Paper, role_results: list[RoleResult], public_role: str) -> str:
        role_outputs = "\n".join(format_role_result(result) for result in role_results)
        prompt = comment_synthesis_prompt(paper, role_outputs, public_role)
        text = self.generate_text(prompt).strip()
        if text:
            return text
        return synthesize_comment_heuristic(paper, role_results, public_role)


def parse_json_object(text: str) -> dict | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def heuristic_role_result(
    role_name: str, paper: Paper, discussion_excerpt: str, raw_text: str = ""
) -> RoleResult:
    text = f"{paper.title}\n{paper.abstract}".lower()
    confidence = 0.42
    probability = 0.5
    weaknesses: list[str] = []
    strengths: list[str] = []

    if any(term in text for term in ["proof", "theorem", "bound", "convergence"]):
        strengths.append("The paper appears to contain theory-relevant claims that need scope checking.")
        if role_name in {"theory_math", "claim_evidence_auditor"}:
            confidence += 0.1
    if any(term in text for term in ["experiment", "benchmark", "dataset", "baseline"]):
        strengths.append("The work appears to include empirical evaluation that can be audited.")
        if role_name in {"experimental_rigor", "benchmark_dataset", "reproducibility"}:
            confidence += 0.1
    if any(term in text for term in ["state-of-the-art", "sota", "outperform"]):
        weaknesses.append("Performance claims should be checked against baseline fairness and variance.")
        probability -= 0.03
    if any(term in text for term in ["novel", "first", "new framework"]):
        weaknesses.append("Novelty claims should be checked against closely related prior work.")
    if paper.github_urls:
        strengths.append("A linked repository may improve reproducibility if it matches the method.")
        probability += 0.03
    if discussion_excerpt:
        confidence += 0.05

    if not strengths:
        strengths.append("The abstract states a potentially relevant ML contribution.")
    if not weaknesses:
        weaknesses.append("The main uncertainty is whether the evidence is strong enough for ICML acceptance.")

    return RoleResult(
        role_name=role_name,
        accept_probability=clamp(probability),
        confidence=clamp(confidence),
        strengths=strengths[:4],
        weaknesses=weaknesses[:4],
        uncertainties=["Full-paper details are needed to calibrate the verdict beyond the abstract."],
        public_points=[*strengths[:2], *weaknesses[:2]],
        fatal_flaws=[],
        questions=["Which experiment, ablation, or analysis most directly supports the central claim?"],
        evidence_refs=[],
        score_factors={
            "soundness": probability,
            "evidence_quality": probability,
            "significance": 0.5,
            "originality": 0.5,
            "clarity": 0.5,
            "reproducibility": 0.55 if paper.github_urls else 0.45,
            "ethics": 0.5,
        },
        raw_text=raw_text,
    )


def format_role_result(result: RoleResult) -> str:
    return (
        f"Role: {result.role_name}\n"
        f"P(accept): {result.accept_probability:.2f}, confidence: {result.confidence:.2f}\n"
        f"Strengths: {'; '.join(result.strengths)}\n"
        f"Weaknesses: {'; '.join(result.weaknesses)}\n"
        f"Fatal flaws: {'; '.join(result.fatal_flaws)}\n"
        f"Uncertainties: {'; '.join(result.uncertainties)}\n"
        f"Questions: {'; '.join(result.questions)}\n"
        f"Evidence refs: {'; '.join(result.evidence_refs)}\n"
        f"Score factors: {json.dumps(result.score_factors, sort_keys=True)}\n"
        f"Public points: {'; '.join(result.public_points)}\n"
    )


def synthesize_comment_heuristic(
    paper: Paper, role_results: list[RoleResult], public_role: str
) -> str:
    strengths = dedupe(point for result in role_results for point in result.strengths)[:3]
    weaknesses = dedupe(point for result in role_results for point in result.weaknesses)[:3]
    fatal_flaws = dedupe(point for result in role_results for point in result.fatal_flaws)[:2]
    uncertainties = dedupe(point for result in role_results for point in result.uncertainties)[:2]
    questions = dedupe(point for result in role_results for point in result.questions)[:2]
    avg_prob = sum(result.accept_probability for result in role_results) / max(1, len(role_results))
    leaning = "weak accept" if avg_prob >= 0.58 else "borderline" if avg_prob >= 0.45 else "weak reject"
    return "\n".join(
        [
            f"I focus on {public_role.replace('_', ' ')} for this paper.",
            "",
            "Summary:",
            f"The paper appears to target {paper.title or 'the stated ML problem'}; my assessment focuses on whether the claims are supported at ICML standard.",
            "",
            "Decision-relevant strengths:",
            *[f"- {item}" for item in strengths],
            "",
            "Decision-relevant concerns:",
            *[f"- {item}" for item in weaknesses],
            *( ["- Potential fatal-flaw candidates: " + "; ".join(fatal_flaws)] if fatal_flaws else [] ),
            "",
            "What would change my assessment:",
            *[f"- {item}" for item in uncertainties],
            *[f"- {item}" for item in questions],
            "",
            "Provisional leaning:",
            f"{leaning}; I would calibrate this around P(accept)={avg_prob:.2f} before reading further discussion and full-paper details.",
        ]
    )


def dedupe(items) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = " ".join(str(item).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def parse_score_factors(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, item in value.items():
        try:
            output[str(key)] = clamp(float(item))
        except (TypeError, ValueError):
            continue
    return output


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return to_jsonable(value.__dict__)
    return value
