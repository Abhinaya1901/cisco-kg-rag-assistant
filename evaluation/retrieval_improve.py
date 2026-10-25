#!/usr/bin/env python3
"""
retrieval_improve.py -- can we retrieve better on the TeleQnA / 3GPP corpus?  No LLM, no Groq tokens.

Compares, on the same 200 questions and the same two gold definitions as tele_retrieval_eval.py:
  1. minilm        current system: all-MiniLM-L6-v2 dense search (your existing chroma_3gpp index)
  2. bm25          keyword search (own implementation, no extra package)
  3. minilm+bm25   reciprocal-rank fusion of 1 and 2 (3GPP text is full of exact terms and acronyms)
  4. bge-small     BAAI/bge-small-en-v1.5 dense search (embeds the 25k passages once, cached in bge_small.npy)
  5. bge+bm25      fusion of 4 and 2
  6. best+rerank   cross-encoder re-ranking of the top 30 of the best first stage
Reports Hit@5, Hit@10, MRR, nDCG@5 with 95% bootstrap intervals and the paired Hit@5 difference to the
current system (wins / ties / losses).  The first run downloads the models (needs internet) and embeds the
passages (about 10-20 minutes on a laptop); later runs reuse the cache.

Run from the teleqna folder:
  python3 -u retrieval_improve.py --n 200
  python3 -u retrieval_improve.py --n 200 --skip-rerank       # if the cross-encoder cannot be downloaded
Writes retrieval_improve_report.md
"""
import argparse, json, math, os, random, re
from collections import Counter
import numpy as np

REPORT = 'retrieval_improve_report.md'
BGE_FILE = 'bge_small.npy'
TOKEN = re.compile(r'[a-z0-9]+')


# --------------------------------------------------------------------------- ranking helpers
def tok(s):
    return TOKEN.findall((s or '').lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tok(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        self.inv = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.inv.setdefault(t, []).append(i)

    def rank(self, query, k=50):
        scores = {}
        for t in set(tok(query)):
            if t not in self.idf:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                s = self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                scores[i] = scores.get(i, 0.0) + s
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def rrf(lists, k=60, top=50):
    sc = {}
    for lst in lists:
        for r, i in enumerate(lst):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:top]]


def metrics_from_rank(r):
    return {'hit5': 1.0 if (r and r <= 5) else 0.0, 'hit10': 1.0 if (r and r <= 10) else 0.0,
            'mrr': (1.0 / r) if (r and r <= 10) else 0.0,
            'ndcg5': (1.0 / math.log2(r + 1)) if (r and r <= 5) else 0.0}


def rank_of(gold, ranked_ids):
    return ranked_ids.index(gold) + 1 if gold in ranked_ids else None


def boot(vals, iters=5000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1]


def evaluate(systems, items, golds):
    """systems: {name: {question_index: [ranked chunk ids]}}; golds: {'A': [...], 'B': [...]}"""
    out = {}
    for tag, gl in golds.items():
        res = {}
        for name, ranks in systems.items():
            per = [metrics_from_rank(rank_of(gl[i], ranks[i])) for i in range(len(items))]
            res[name] = per
        out[tag] = res
    return out


def report(ev, items, names, base):
    L = ['# Retrieval improvement on TeleQnA (3GPP Release 18)', '',
         '%d questions, query = question text only, no LLM. Gold A = best-overlap passage among the top 10 retrieved '
         'for question + answer; Gold B = best-overlap passage in the whole corpus. Paired difference is on Hit@5 '
         'against "%s".' % (len(items), base), '']
    for tag in ('A', 'B'):
        L += ['## Gold %s' % tag, '', '| System | Hit@5 | 95%% interval | Hit@10 | MRR | nDCG@5 | Hit@5 diff vs %s | W / T / L |' % base,
              '|---|---|---|---|---|---|---|---|']
        b5 = [m['hit5'] for m in ev[tag][base]]
        for n in names:
            per = ev[tag][n]
            h5 = [m['hit5'] for m in per]
            lo, hi = boot(h5)
            d = [x - y for x, y in zip(h5, b5)]
            w, l = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
            dl, dh = boot(d) if n != base else (0, 0)
            L.append('| %s | %.3f | %.3f to %.3f | %.3f | %.3f | %.3f | %s | %s |' % (
                n, np.mean(h5), lo, hi, np.mean([m['hit10'] for m in per]), np.mean([m['mrr'] for m in per]),
                np.mean([m['ndcg5'] for m in per]),
                '-' if n == base else '%+.3f (%+.3f to %+.3f)' % (np.mean(d), dl, dh),
                '-' if n == base else '%d / %d / %d' % (w, len(d) - w - l, l)))
        L.append('')
    L += ['A difference is real only when its interval excludes 0. Gold passages are a word-overlap proxy; '
          'a near-duplicate passage counts as a miss.']
    return '\n'.join(L)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--skip-rerank', action='store_true')
    ap.add_argument('--skip-bge', action='store_true')
    a = ap.parse_args()

    import evaluation_tele as T
    from sentence_transformers import SentenceTransformer
    pipe = T.TelePipeline()
    items = T.build_items(pipe, a.n)
    chunks = pipe.chunks
    ids = [c['id'] for c in chunks]
    texts = [c['text'] for c in chunks]
    idx = {c: i for i, c in enumerate(ids)}
    n = len(items)

    # gold definitions (same as tele_retrieval_eval.py)
    gold_a = [it['gold_chunk_id'] for it in items]
    chunk_sets = [set(T.toks(t)) for t in texts]
    gold_b = []
    for it in items:
        ref = set(T.toks(it['reference']))
        best, g = -1.0, None
        for i, ts in enumerate(chunk_sets):
            ov = len(ref & ts) / max(1, len(ref))
            if ov > best:
                best, g = ov, ids[i]
        gold_b.append(g)

    qs = [it['question'] for it in items]
    print('Building BM25 index...', flush=True)
    bm = BM25(texts)
    bm_ranks = [bm.rank(q, 50) for q in qs]
    mini_ranks = [[idx[c['id']] for c in pipe.retrieve(q, 50)] for q in qs]
    systems_idx = {'minilm': mini_ranks, 'bm25': bm_ranks, 'minilm+bm25': [rrf([m, b]) for m, b in zip(mini_ranks, bm_ranks)]}

    best_name = 'minilm+bm25'
    if not a.skip_bge:
        print('Loading BGE-small and embedding passages (cached after the first run)...', flush=True)
        bge = SentenceTransformer('BAAI/bge-small-en-v1.5')
        if os.path.exists(BGE_FILE):
            emb = np.load(BGE_FILE)
        else:
            emb = bge.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            np.save(BGE_FILE, emb)
        qv = bge.encode(['Represent this sentence for searching relevant passages: ' + q for q in qs],
                        normalize_embeddings=True)
        sims = qv @ emb.T
        bge_ranks = [list(np.argsort(-sims[i])[:50]) for i in range(n)]
        systems_idx['bge-small'] = bge_ranks
        systems_idx['bge+bm25'] = [rrf([g, b]) for g, b in zip(bge_ranks, bm_ranks)]

    if not a.skip_rerank:
        print('Loading cross-encoder and re-ranking...', flush=True)
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # choose the best first stage by Hit@5 on gold A
        def h5(r):
            return np.mean([1.0 if gold_a[i] in [ids[j] for j in r[i][:5]] else 0.0 for i in range(n)])
        best_name = max(systems_idx, key=lambda k: h5(systems_idx[k]))
        stage = systems_idx[best_name]
        rer = []
        for i in range(n):
            cand = stage[i][:30]
            sc = ce.predict([(qs[i], texts[j][:1000]) for j in cand])
            rer.append([cand[k] for k in np.argsort(-np.array(sc))])
        systems_idx[best_name + '+rerank'] = rer

    systems = {k: {i: [ids[j] for j in v[i]] for i in range(n)} for k, v in systems_idx.items()}
    ev = evaluate(systems, items, {'A': gold_a, 'B': gold_b})
    text = report(ev, items, list(systems.keys()), 'minilm')
    open(REPORT, 'w').write(text)
    print(text)
    print('\nWrote %s' % REPORT)


if __name__ == '__main__':
    main()
