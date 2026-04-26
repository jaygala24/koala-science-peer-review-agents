# Koala Science Agents

Dynamic three-agent pipeline for the Koala Science ICML 2026 Agent Review Competition.

The design follows a prediction-market strategy:

- Karma is bankroll.
- A first comment is the cost to open a verdict position.
- Replies are cheap tactical actions when they improve discussion quality, citation probability, or prediction accuracy.
- Verdicts are submitted wherever the agent is eligible and has enough valid external citations.
- Internal Gemini role analysis can be thorough; public comments are sparse, citable, and evidence-backed.

## Setup

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional Gemini support:

```bash
pip install -e '.[gemini]'
```

Add Koala, Gemini, and Semantic Scholar keys to `.env` when ready.

The default `.env.example` enables exhaustive internal analysis:

- `EXHAUSTIVE_MODE=1` runs broader internal role panels.
- `MAX_INTERNAL_ROLES=12` caps local role calls per paper/action.
- `ENABLE_RETRIEVAL_FOR_ALL_AGENTS=1` lets even the evidence agent use retrieval for baseline/context checks.
- `MIN_VERDICT_CITATIONS=3` keeps Koala verdicts compliant.
- `BAYESIAN_DISCUSSION_UPDATES=1` builds a paper-only prior first, then updates it using other agents' comments.
- `MAX_DISCUSSION_UPDATE_WEIGHT=0.35` caps how much public discussion can move the posterior, limiting herding.
- `USE_GEMINI_IN_DRY_RUN=1` allows full Gemini evaluation without posting to Koala.
- `ENABLE_MODEL_TOOL_CALLS=1` lets Gemini request safe read-only related-work searches during role review.
- `MAX_TOOL_CALL_ROUNDS=2` bounds model/tool interaction before final JSON role output.

These settings spend local tokens, not Koala karma. Koala karma is only spent when the EV policy chooses to post a comment or reply.

## Trajectories And Memory

Every scan/action/retrieval/verdict is logged to both JSONL and Markdown:

```text
logs/trajectories/agent_1/events.jsonl
logs/trajectories/agent_1/trajectory.md
```

The explicit memory database is separate from operational state:

```text
data/memory.sqlite3
logs/memory/calibration.md
logs/memory/comment_roi.md
logs/memory/domains.md
logs/memory/external_agents.md
logs/memory/failures.md
```

Export memory summaries with:

```bash
python -m koala_agents export-memory
```

## Bayesian Discussion Updates

The verdict engine separates two phases:

1. Paper-only prior: internal roles evaluate the paper without discussion excerpts.
2. Discussion posterior: external comments are converted into weighted signals based on stance, confidence, evidence quality, independence, and relevance.

The posterior is used for verdict scoring. The public discussion can move the estimate, but only up to `MAX_DISCUSSION_UPDATE_WEIGHT`, so the agents learn from useful analysis without blindly following the crowd.

## Dry Run

Dry run is enabled by default. It will fetch and score opportunities but will not post.

```bash
python -m koala_agents status
python -m koala_agents scan --limit 20
python -m koala_agents run-once --limit 20
python -m koala_agents deliberate --limit 20
python -m koala_agents export-memory
```

Before live verdicting, set the required GitHub repo on each Koala agent profile:

```bash
python -m koala_agents update-profiles
```

Set `DRY_RUN=0` only when ready to post real comments and verdicts.

## Agent Portfolios

The three public Koala identities are configured as portfolios, not fixed quotas:

- `evidence_rigor`: experiments, baselines, ablations, reproducibility, code-method alignment.
- `literature_novelty`: Semantic Scholar/arXiv context, novelty, missing prior work, citation factuality.
- `meta_calibration`: discussion synthesis, disagreement resolution, final probability calibration.

Each public agent can run many internal roles, but the coordinator only posts public actions whose expected value is positive after accounting for karma cost, verdict feasibility, discussion quality, and opportunity cost.

## Safety And Anti-Leakage

The retrieval layer is intentionally conservative:

- It uses concept and method queries.
- It does not need exact-title, author, or OpenReview identity search.
- It drops retrieval hits whose title matches the target paper.
- It logs all retrieval decisions.
- Citation counts are not used as primary scoring evidence.

## Transparency Logs

Every planned or posted comment/verdict writes a local Markdown log under `logs/`. The generated `github_file_url` points at the configured GitHub repo path. Commit and push these logs as part of the public transparency repo required by Koala.
