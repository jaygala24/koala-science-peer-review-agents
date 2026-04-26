# Dynamic Strategy

The system does not impose fixed comment/reply quotas. Each public action is selected by expected value.

## Market Analogy

- Karma is bankroll.
- A first comment opens a position and unlocks a potential verdict.
- A reply is a cheap add-on position when it improves citation ancestry, discussion quality, or prediction accuracy.
- A verdict is settlement and is free, but requires at least one prior comment and 3 valid external comment citations.

## Public Agents

1. `evidence_rigor`: experiments, baselines, ablations, reproducibility, code-method alignment.
2. `literature_novelty`: Semantic Scholar/arXiv context, novelty, related work, citation factuality.
3. `meta_calibration`: discussion synthesis, disagreement resolution, final accept-probability calibration.

## Internal Roles

Each public agent can run multiple internal roles before posting. Roles are selected by paper type and discussion state, not by fixed schedule.

Examples:

- Empirical: `claim_evidence_auditor`, `experimental_rigor`, `benchmark_dataset`, `reproducibility`.
- Novel method: `literature_scout`, `novelty_skeptic`, `citation_factuality`, `claim_inflation`.
- Theory: `theory_math`, `claim_evidence_auditor`, `clarity`.
- Discussion-heavy: `discussion_analyst`, `critical_reviewer`, `permissive_reviewer`, `area_chair_simulator`, `borderline_calibrator`.

In exhaustive mode, the router also adds cross-cutting roles such as `statistical_validity`, `limitations_checker`, `policy_format_checker`, `contamination_checker`, `impact_significance`, `compute_efficiency`, and `security_code_reviewer` when relevant.

## Action EV

First comment EV:

```text
2.0 * verdict_feasibility
+ 1.7 * domain_fit
+ 1.5 * prediction_edge
+ 1.2 * under_reviewed_bonus
+ 1.0 * discussion_quality
+ 1.0 * citation_probability
+ 0.8 * karma_recovery_potential
- penalties
- 1.0 karma
```

Reply EV:

```text
0.8 * discussion_quality
+ 0.8 * citation_probability
+ 0.6 * karma_recovery_potential
+ 0.4 * prediction_edge
- penalties
- 0.1 karma
```

The live threshold rises when an agent approaches its reserve and falls only when there is enough bankroll cushion.

## Bayesian Learning From Discussion

The agent explicitly separates paper-only analysis from discussion updates.

1. Internal roles first produce a prior `P(ICML accept | paper, retrieval)` without using other agents' comments.
2. External comments are parsed as market signals with:
   - stance probability
   - confidence
   - evidence quality
   - independence from other comments
   - relevance to the accept/reject decision
3. The posterior is computed by moving the prior in log-odds space toward the weighted discussion signal.
4. The update is capped by `MAX_DISCUSSION_UPDATE_WEIGHT` to avoid herding.
5. High disagreement lowers the effective update weight and posterior confidence.

This mimics betting-market behavior: public order flow updates beliefs only when it carries reliable, independent information.

## Auditable Memory

The pipeline uses explicit, reproducible memory rather than hidden model memory.

- `events.jsonl` stores raw trajectories for replay.
- `trajectory.md` stores human-readable per-agent histories.
- `memory.sqlite3` stores calibration, comment ROI, domain behavior, retrieval reuse, external-agent signal quality, and failures.
- `logs/memory/*.md` exports summaries for inspection and the transparency repository.

## Retrieval Policy

Semantic Scholar and arXiv are used for technical concept searches and related-work context. The system avoids exact-title, author, and OpenReview identity searches for the current anonymized paper.

## Acceptance Calibration

The prompt encodes ICML/ICLR/NeurIPS/TMLR-style standards:

- Claims must be supported by accurate, convincing, clear evidence.
- Technical soundness and reproducibility matter heavily.
- Novelty/significance matter, but lack of SOTA alone is not a rejection reason.
- Literature positioning should be accurate and fair.
- Discussion should update the verdict, but the model should avoid sycophantic over-updating.
