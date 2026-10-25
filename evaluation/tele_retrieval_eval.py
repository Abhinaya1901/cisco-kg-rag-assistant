#!/usr/bin/env python3
"""
tele_retrieval_eval.py -- retrieval quality on the TeleQnA / 3GPP Release 18 corpus.
Uses NO LLM and NO Groq tokens (local embedding model + local index only).

For each question (question text only is the query; the answer is never used) it measures:
  * vector retrieval: Precision@5, Recall@5, MRR, nDCG@5, Hit@5, Hit@10 against a gold passage
  * hybrid retrieval: whether the structural graph (cross-reference / continuation passages, the same
    expansion the generator receives) adds the gold passage when vector search missed it
  * 95% bootstrap intervals for the hit rates and MRR

Two gold-passage definitions are reported, so a reader can check the result does not depend on one choice:
  A "retrieval-derived": the top-10 passage (for question + answer) with the highest word overlap with the
                          dataset explanation (what tele_items.json stores)
  B "corpus-wide":        the passage with the highest word overlap with the explanation among ALL passages
                          (does not depend on any retrieval)
Only questions whose best overlap is >= 0.6 are kept (the corpus actually covers them).

Run from the teleqna folder (needs evaluation_v2.py, evaluation_v3.py, evaluation_tele.py and the corpus files):
  python3 -u tele_retrieval_eval.py --n 200
Writes tele_retrieval_report.md and tele_retrieval_per_question.csv
"""
import argparse, csv, json, math, random
import numpy as np

REPORT = 'tele_retrieval_report.md'
CSV_FILE = 'tele_retrieval_per_question.csv'


def metrics(rank_of_gold):
    """rank_of_gold: 1-based rank of the gold passage in the ranked list, or None if absent from the top 10."""
    rel = [1 if rank_of_gold == i + 1 else 0 for i in range(10)]
    r = rank_of_gold
    return {'p5': sum(rel[:5]) / 5.0,
            'r5': 1.0 if (r and r <= 5) else 0.0,
            'mrr': (1.0 / r) if r else 0.0,
            'ndcg5': (1.0 / math.log2(r + 1)) if (r and r <= 5) else 0.0,
            'hit5': 1.0 if (r and r <= 5) else 0.0,
            'hit10': 1.0 if r else 0.0}


def graph_added_ids(pipe, top_ids, max_facts, k_gen=5):
    """Same expansion as TelePipeline.graph_facts, but returns passage ids."""
    shown = set(top_ids[:k_gen])
    out = []
    for seed in top_ids[:3]:
        if seed not in pipe.by_id:
            continue
        cand = list(pipe.xref.get(seed, []))
        if seed in pipe.nxt:
            cand.append(pipe.nxt[seed])
        for tid in cand:
            if tid in shown or tid in out:
                continue
            out.append(tid)
            if len(out) >= max_facts:
                return out
    return out


def boot_ci(vals, iters=5000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return None
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1]


def summarise(rows, key):
    out = {}
    for m in ('p5', 'r5', 'mrr', 'ndcg5', 'hit5', 'hit10'):
        v = [r[key][m] for r in rows]
        out[m] = (float(np.mean(v)), boot_ci(v))
    return out


def run(pipe, items, overlap, toks, max_facts):
    all_chunks = pipe.all_chunks()
    chunk_toks = [(c['id'], set(toks(c['text']))) for c in all_chunks]
    rows = []
    for it in items:
        q = it['question']
        ref_toks = toks(it['reference'])
        ranked = [c['id'] for c in pipe.retrieve(q, 10)]
        # gold B: corpus-wide best overlap
        refset = set(ref_toks)
        best_b, gold_b = -1.0, None
        for cid, ts in chunk_toks:
            ov = len(refset & ts) / max(1, len(refset))
            if ov > best_b:
                best_b, gold_b = ov, cid
        gold_a = it['gold_chunk_id']
        added = graph_added_ids(pipe, ranked, max_facts)
        row = {'id': it['id'], 'question': q, 'gold_a': gold_a, 'gold_b': gold_b, 'overlap_b': round(best_b, 3),
               'graph_added': len(added)}
        for tag, gold in (('A', gold_a), ('B', gold_b)):
            rank = ranked.index(gold) + 1 if gold in ranked else None
            row[tag] = metrics(rank)
            row[tag + '_rank'] = rank
            in_graph = (gold in added) and not (rank and rank <= 5)
            row[tag + '_graph_rescue'] = 1.0 if in_graph else 0.0
            row[tag + '_hit5_hybrid'] = 1.0 if ((rank and rank <= 5) or gold in added) else 0.0
        rows.append(row)
    return rows


def write_outputs(rows, n_total):
    L = ['# Retrieval quality: TeleQnA (3GPP Release 18 subset)', '',
         'Questions: %d (covered by the corpus, closed-style removed). Query = question text only. '
         'No LLM and no judge: gold passages are defined by word overlap with the dataset explanation.' % len(rows), '',
         'Gold A = best-overlap passage among the top 10 retrieved for question + answer. '
         'Gold B = best-overlap passage among all %d passages (independent of retrieval).' % n_total, '']
    for tag, name in (('A', 'Gold A (retrieval-derived)'), ('B', 'Gold B (corpus-wide)')):
        s = summarise(rows, tag)
        L += ['## Vector retrieval, ' + name, '', '| Metric | Value | 95% interval |', '|---|---|---|']
        labels = {'p5': 'Precision@5', 'r5': 'Recall@5 (hit)', 'mrr': 'MRR', 'ndcg5': 'nDCG@5',
                  'hit5': 'Hit@5', 'hit10': 'Hit@10'}
        for m in ('p5', 'r5', 'mrr', 'ndcg5', 'hit5', 'hit10'):
            v, ci = s[m]
            L.append('| %s | %.3f | %.3f to %.3f |' % (labels[m], v, ci[0], ci[1]))
        h5 = [r[tag]['hit5'] for r in rows]
        hh = [r[tag + '_hit5_hybrid'] for r in rows]
        resc = [r[tag + '_graph_rescue'] for r in rows]
        diff = [b - a for a, b in zip(h5, hh)]
        L += ['', '| Hybrid (vector top-5 + structural graph passages) | Value | 95% interval |', '|---|---|---|',
              '| Gold present in top 5 (vector only) | %.3f | %.3f to %.3f |' % ((np.mean(h5),) + boot_ci(h5)),
              '| Gold present in top 5 or graph passages | %.3f | %.3f to %.3f |' % ((np.mean(hh),) + boot_ci(hh)),
              '| Questions where the graph supplied a gold passage that vector missed | %d of %d | |' % (int(sum(resc)), len(rows)),
              '', '']
    L += ['Mean passages added by the graph per question: %.2f.' % np.mean([r['graph_added'] for r in rows]), '',
          'Reading guide: the graph can only help retrieval if the gold passage is a cross-referenced or '
          'continuation neighbour of a top-3 passage. The "supplied a gold passage" count shows how often that happened.',
          'Limits: gold passages are a word-overlap proxy, not human judgements; a near-duplicate passage counts as a miss.']
    open(REPORT, 'w').write('\n'.join(L))
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'question', 'gold_A', 'rank_A', 'gold_B', 'rank_B', 'overlap_B', 'graph_added',
                    'graph_rescue_A', 'graph_rescue_B'])
        for r in rows:
            w.writerow([r['id'], r['question'][:200], r['gold_a'], r['A_rank'], r['gold_b'], r['B_rank'],
                        r['overlap_b'], r['graph_added'], r['A_graph_rescue'], r['B_graph_rescue']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    import evaluation_tele as T
    pipe = T.TelePipeline()
    items = T.build_items(pipe, a.n)
    rows = run(pipe, items, T.overlap, T.toks, T.MAX_FACTS)
    write_outputs(rows, len(pipe.chunks))
    print(open(REPORT).read())
    print('\nWrote %s and %s' % (REPORT, CSV_FILE))


if __name__ == '__main__':
    main()
