# Automated Evaluation Report

- Questions evaluated: 30 (hand-written: 18, synthetic: 12)
- Answer-generation model: openai/gpt-oss-120b | Judge model: openai/gpt-oss-20b
- Scores are 1-5 (higher is better); mean ± standard deviation over questions.
- Questions with failed parts (excluded from the averages, not counted as zero): none

## 1. Retrieval quality

**Hand-written questions** (passage relevance judged by the LLM, top-10 judged, n=18)

| Precision@5 | Recall@5 | MRR | nDCG@5 | Hit@5 |
|---|---|---|---|---|
| 0.289 | 0.757 | 0.547 | 0.516 | 0.833 |

| Category | n | Precision@5 | Recall@5 | MRR | nDCG@5 | Hit@5 |
|---|---|---|---|---|---|---|
| factual | 6 | 0.267 | 0.653 | 0.386 | 0.381 | 0.833 |
| relational | 8 | 0.325 | 0.863 | 0.692 | 0.691 | 1.000 |
| troubleshooting | 4 | 0.250 | 0.650 | 0.500 | 0.367 | 0.500 |

**Synthetic questions** (objective: the source chunk must be retrieved, n=12)

| Hit@5 | Hit@10 | MRR | nDCG@5 |
|---|---|---|---|
| 0.833 | 0.917 | 0.747 | 0.761 |

Note: hybrid and vector-only use the same top-k vector retrieval; the knowledge graph adds extra facts on top, so retrieval metrics describe the vector stage.

## 2. Answer quality (ablation: vector-only vs KG-only vs hybrid)

| System | n | Faithfulness | Relevance | Correctness* | Hallucination rate** |
|---|---|---|---|---|---|
| Vector-only | 30 | 4.40 ± 1.19 | 4.67 ± 1.03 | 4.17 ± 1.59 | 9.3% (161 claims) |
| KG-only | 30 | 2.53 ± 1.94 | 2.33 ± 1.84 | 1.00 ± 0.00 | 54.1% (37 claims) |
| Hybrid (KG+Vector) | 30 | 4.17 ± 1.12 | 4.87 ± 0.51 | 4.50 ± 1.24 | 21.4% (187 claims) |

\* Correctness is measured against a reference answer, so it exists only for synthetic questions.  
\** Share of individual claims that the judge found unsupported by the retrieved context (lower is better).

**By question category, hand-written set (Faithfulness / Relevance)**

| Category | Vector-only | KG-only | Hybrid (KG+Vector) |
|---|---|---|---|
| factual | 4.33 / 5.00 | 3.67 / 3.67 | 4.33 / 5.00 |
| relational | 4.25 / 4.62 | 3.25 / 3.50 | 4.25 / 5.00 |
| troubleshooting | 4.00 / 5.00 | 3.00 / 2.00 | 4.50 / 5.00 |

## 3. Hybrid vs vector-only (paired, per question)

| Metric | n | Mean difference (hybrid − vector) | 95% bootstrap CI | Wins / ties / losses |
|---|---|---|---|---|
| Faithfulness | 30 | -0.23 | [-0.67, +0.20] | 4 / 18 / 8 |
| Relevance | 30 | +0.20 | [+0.00, +0.47] | 3 / 27 / 0 |
| Correctness | 12 | +0.33 | [+0.00, +0.83] | 2 / 10 / 0 |

A confidence interval that excludes 0 means the difference is unlikely to be chance.

## 4. Efficiency and knowledge-graph coverage

- Questions where the knowledge graph supplied at least one fact: 97% (mean 5.0 facts per question)
- Median vector retrieval time: 146 ms | median KG traversal time: 1.4 ms

| System | Median generation latency (s) | Mean tokens per answer (prompt+completion) |
|---|---|---|
| Vector-only | 0.9 | 1529 |
| KG-only | 0.5 | 275 |
| Hybrid (KG+Vector) | 1.0 | 1637 |

## 5. Validity notes and limitations

- Judging is done by an LLM. See judge_audit.csv: a human spot-check of a sample, summarised by `--agreement`.
- Generator and judge are from the same model family, which can favour similar phrasing. Different model sizes are used for the two roles.
- The generator and the judge see identical context text (no silent truncation of the knowledge-graph facts).
- Synthetic questions are written from a single chunk, so they favour lexical/vector retrieval; the hand-written set contains the relational and multi-document questions where the graph should help.
- The synthetic gold-chunk metric is strict: a near-duplicate chunk of the same content counts as a miss.
- Small sample sizes: use the confidence intervals, not just the means.
