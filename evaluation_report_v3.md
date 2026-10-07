# Additional Evidence Report (controlled comparison, objective similarity, second judge)

- Questions: 30 (hand-written 18, synthetic 12). Scores 1-5, mean ± SD.
- Main judge: openai/gpt-oss-20b | Second judge: openai/gpt-oss-120b

## 1. Controlled ablation

"Hybrid, strict prompt" uses the production hybrid prompt plus the rule "use ONLY the context", so it differs from vector-only in one thing only: the knowledge-graph facts.

| System | n | Faithfulness | Relevance | Correctness* | Hallucination rate** |
|---|---|---|---|---|---|
| Vector-only | 30 | 4.40 ± 1.19 | 4.67 ± 1.03 | 4.17 ± 1.59 | 9.3% (161 claims) |
| KG-only | 30 | 2.53 ± 1.94 | 2.33 ± 1.84 | 1.00 ± 0.00 | 54.1% (37 claims) |
| Hybrid, production prompt | 30 | 4.17 ± 1.12 | 4.87 ± 0.51 | 4.50 ± 1.24 | 21.4% (187 claims) |
| Hybrid, strict prompt (controlled) | 17 | 4.88 ± 0.49 | 4.88 ± 0.49 | - | 2.4% (84 claims) |

\* only for the generated questions (they have a reference answer). \** unsupported claims / all claims.

**By category, hand-written set (Faithfulness / Relevance)**

| Category | Vector-only | KG-only | Hybrid, production prompt | Hybrid, strict prompt (controlled) |
|---|---|---|---|---|
| factual | 4.33 / 5.00 | 3.67 / 3.67 | 4.33 / 5.00 | 5.00 / 4.67 |
| relational | 4.25 / 4.62 | 3.25 / 3.50 | 4.25 / 5.00 | 4.75 / 5.00 |
| troubleshooting | 4.00 / 5.00 | 3.00 / 2.00 | 4.50 / 5.00 | 5.00 / 5.00 |

## 2. Paired comparisons (difference per question, 95% bootstrap CI)

| Comparison | Metric | n | Mean diff | 95% CI | W / T / L | CI excludes 0? |
|---|---|---|---|---|---|---|
| Hybrid, strict prompt (controlled) − Vector-only | Faithfulness | 17 | +0.71 | [+0.00, +1.41] | 5 / 11 / 1 | no |
| Hybrid, strict prompt (controlled) − Vector-only | Relevance | 17 | +0.06 | [-0.35, +0.53] | 1 / 15 / 1 | no |
| Hybrid, production prompt − Vector-only | Faithfulness | 30 | -0.23 | [-0.67, +0.20] | 4 / 18 / 8 | no |
| Hybrid, production prompt − Vector-only | Relevance | 30 | +0.20 | [+0.00, +0.47] | 3 / 27 / 0 | no |
| Hybrid, production prompt − Vector-only | Correctness | 12 | +0.33 | [+0.00, +0.83] | 2 / 10 / 0 | no |
| Hybrid, strict prompt (controlled) − Hybrid, production prompt | Faithfulness | 17 | +0.47 | [+0.00, +1.06] | 3 / 14 / 0 | no |
| Hybrid, strict prompt (controlled) − Hybrid, production prompt | Relevance | 17 | -0.12 | [-0.35, +0.00] | 0 / 16 / 1 | no |

A difference is called real only when the CI excludes 0.

## 3. Objective similarity to the reference answer (generated questions, no LLM judge)

| System | n | Reference-word recall | Embedding cosine |
|---|---|---|---|
| Vector-only | 0 | - | - |
| KG-only | 0 | - | - |
| Hybrid, production prompt | 0 | - | - |
| Hybrid, strict prompt (controlled) | 0 | - | - |

Reference-word recall = share of the reference answer's content words found in the answer. Cosine = MiniLM sentence-embedding similarity. Both are rough, objective cross-checks.

## 4. Second-judge agreement (replaces a human check)

No second-judge scores yet.

## 5. How to read this honestly

- Claims are supported only where the 95% CI excludes 0. Small samples can hide real effects and can also produce lucky ones.
- The generated questions come from one chunk each, so they favour plain vector search; the relational hand-written questions are the fairer test of the graph.
- All judging is automatic. The agreement table shows the two judges are consistent, which is weaker evidence than human agreement.
- Results come from one model family (gpt-oss) and one 18+12 question set on Catalyst 9300 documentation.
