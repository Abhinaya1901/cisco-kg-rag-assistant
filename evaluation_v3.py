#!/usr/bin/env python3
"""
evaluation_v3.py -- extra evidence on top of evaluation_v2.py.  Run evaluation_v2.py first.

Run from the project root (same folder as evaluation_v2.py):

  python3 -u evaluation_v3.py            # does everything below, resumable
  python3 -u evaluation_v3.py --report   # only rebuild evaluation_report_v3.md

What it adds (reads eval_v2_results.json, writes eval_v3_results.json, never touches the v2 file):
  1. Controlled hybrid run ("hybrid_strict"): the SAME hybrid prompt as production, plus one extra
     rule "use ONLY the context". This removes the prompt difference that made hybrid look worse than
     vector-only, so the comparison changes one thing only: the knowledge-graph facts.
  2. Objective reference-similarity (generated questions only, no LLM judge involved):
     share of reference-answer words found in the answer, and embedding cosine similarity.
  3. Second-judge check: a different model re-scores a sample of answers; we report how often the
     two judges agree (a no-human substitute for a validity check).
"""
import os
import sys
import re
import json
import math
import random
import argparse
import statistics
import copy
from collections import Counter

import evaluation_v2 as ev

V3_FILE = 'eval_v3_results.json'
REPORT3 = 'evaluation_report_v3.md'
MODES3 = ['vector', 'kg', 'hybrid', 'hybrid_strict']
LABEL3 = {'vector': 'Vector-only', 'kg': 'KG-only',
          'hybrid': 'Hybrid, production prompt',
          'hybrid_strict': 'Hybrid, strict prompt (controlled)'}
JUDGE2_PREF = ['meta-llama/llama-4-scout-17b-16e-instruct', 'qwen/qwen3-32b', 'openai/gpt-oss-120b']
JUDGE2_MODES = ['vector', 'hybrid_strict']

STOP = set('the a an and or of to in on for is are was were be been it its this that with as by at from can '
           'will you your if not no do does which what when how also any all use used using than then there '
           'these those they their have has had may must should would could into only more most such other '
           'each per via'.split())


class LLMStrip(ev.LLM):
    """Same wrapper, but removes <think>...</think> blocks some models put in the answer."""

    def call(self, model, prompt, **kw):
        out = super().call(model, prompt, **kw)
        if out.get('text'):
            t = re.sub(r'<think>.*?</think>', '', out['text'], flags=re.S).strip()
            out['text'] = t or None
        return out


class Pipe3(ev.Pipeline):
    def chunks_by_ids(self, ids):
        res = self.col.get(ids=list(ids), include=['documents'])
        m = dict(zip(res['ids'], res['documents']))
        return [{'id': i, 'text': m.get(i, ''), 'meta': None} for i in ids if i in m]


# --------------------------------------------------------------------------- objective similarity
def content_tokens(s):
    return [t for t in re.findall(r'[a-z0-9]+', (s or '').lower()) if len(t) > 2 and t not in STOP]


def reference_recall(answer, reference):
    """Share of the reference answer's content words that appear in the answer (0-1)."""
    ref = Counter(content_tokens(reference))
    if not ref:
        return None
    ans = Counter(content_tokens(answer))
    hit = sum(min(c, ans[t]) for t, c in ref.items())
    return hit / sum(ref.values())


def cosine(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv) if nu and nv else None


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else None


# --------------------------------------------------------------------------- the three steps
def strict_prompt(q, chunks, facts):
    ct, ft = ev.chunks_text(chunks), ev.facts_text(facts)
    return ('You are a network engineering assistant. Answer the question using ONLY the document excerpts and '
            'knowledge graph relationships below. If they do not contain the answer, say so. Do not use outside '
            'knowledge.\n\nDOCUMENT EXCERPTS:\n%s\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n%s\n\nQuestion: %s\n\n'
            'Answer in plain conversational paragraphs. Do NOT use markdown tables, headers, or bullet '
            'checklists.' % (ct, ft, q))


def strict_done(rec):
    m = rec['modes'].get('hybrid_strict')
    return bool(m) and bool(m.get('answer')) and bool(m.get('judge'))


def do_strict(rec, pipe, llm):
    q = rec['question']
    chunks = pipe.chunks_by_ids(rec['retrieved_ids'][:ev.TOP_K_GEN])
    facts = rec.get('graph_facts', [])
    ctx = ev.build_prompt_and_context('hybrid', q, chunks, facts)[1]
    out = llm.call(rec['gen_model'], strict_prompt(q, chunks, facts), max_tokens=ev.GEN_MAX_TOKENS)
    m = {'answer': out['text'], 'gen_tokens': out['tokens'], 'gen_latency_s': out['latency'],
         'gen_error': out['error'], 'judge': None, 'context_snippet': ctx[:600]}
    if out['text']:
        m['judge'] = ev.judge_answer(llm, rec['judge_model'], q, ctx, out['text'], rec.get('reference'))
    rec['modes']['hybrid_strict'] = m


def do_similarity(rec, pipe):
    ref = rec.get('reference')
    if not ref:
        return
    for mode, m in rec['modes'].items():
        if m.get('answer') and 'sim' not in m:
            cos = None
            try:
                vecs = pipe.embed.encode([m['answer'], ref]).tolist()
                cos = cosine(vecs[0], vecs[1])
            except Exception as e:  # noqa
                print('    [similarity skipped: %s]' % str(e)[:80], flush=True)
            m['sim'] = {'ref_recall': reference_recall(m['answer'], ref), 'cos': cos}


def pick_sample(results, n, seed=7):
    ok = [r for r in results if all(ev.judged(r['modes'].get(md)) for md in JUDGE2_MODES)]
    random.Random(seed).shuffle(ok)
    return [r['id'] for r in ok[:n]]


def do_judge2(rec, pipe, llm2, judge2_model):
    q = rec['question']
    chunks = pipe.chunks_by_ids(rec['retrieved_ids'][:ev.TOP_K_GEN])
    facts = rec.get('graph_facts', [])
    for mode in JUDGE2_MODES:
        m = rec['modes'][mode]
        if m.get('judge2'):
            continue
        ctx = ev.build_prompt_and_context('hybrid' if mode == 'hybrid_strict' else mode, q, chunks, facts)[1]
        j = ev.judge_answer(llm2, judge2_model, q, ctx, m['answer'], rec.get('reference'))
        if j:
            j['model'] = judge2_model
        m['judge2'] = j


def save3(results):
    tmp = V3_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, V3_FILE)


def run_v3(results, pipe, llm, llm2, judge2_model, sample_n, limit=0):
    todo = [r for r in results if not strict_done(r)]
    if limit:
        todo = todo[:limit]
    for n, rec in enumerate(todo, 1):
        print('\n[strict %d/%d] %s: %s' % (n, len(todo), rec['id'], rec['question'][:70]), flush=True)
        do_strict(rec, pipe, llm)
        j = rec['modes']['hybrid_strict'].get('judge')
        print('  hybrid_strict F=%s R=%s' % (j['faithfulness'] if j else 'x', j['relevance'] if j else 'x'), flush=True)
        save3(results)
    for rec in results:
        do_similarity(rec, pipe)
    save3(results)
    ids = pick_sample(results, sample_n)
    need = [r for r in results if r['id'] in ids and any(not r['modes'][md].get('judge2') for md in JUDGE2_MODES)]
    for n, rec in enumerate(need, 1):
        print('\n[second judge %d/%d] %s' % (n, len(need), rec['id']), flush=True)
        do_judge2(rec, pipe, llm2, judge2_model)
        save3(results)
    return results


# --------------------------------------------------------------------------- report
def sim_vals(results, mode, key):
    return [r['modes'][mode]['sim'][key] for r in results
            if mode in r['modes'] and r['modes'][mode].get('sim') and r['modes'][mode]['sim'].get(key) is not None]


def paired_sim(results, key, a, b, iters=5000, seed=0):
    diffs = []
    for r in results:
        ma, mb = r['modes'].get(a), r['modes'].get(b)
        if ma and mb and ma.get('sim') and mb.get('sim') and ma['sim'].get(key) is not None and mb['sim'].get(key) is not None:
            diffs.append(ma['sim'][key] - mb['sim'][key])
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return {'n': n, 'mean': sum(diffs) / n, 'lo': means[int(0.025 * iters)], 'hi': means[int(0.975 * iters) - 1]}


def judge_agreement(results, key):
    pairs = []
    for r in results:
        for mode in JUDGE2_MODES:
            m = r['modes'].get(mode) or {}
            if ev.judged(m) and m.get('judge2') and m['judge'].get(key) is not None and m['judge2'].get(key) is not None:
                pairs.append((m['judge'][key], m['judge2'][key]))
    if not pairs:
        return None
    n = len(pairs)
    return {'n': n, 'exact': sum(1 for a, b in pairs if a == b) / n,
            'within1': sum(1 for a, b in pairs if abs(a - b) <= 1) / n,
            'mae': sum(abs(a - b) for a, b in pairs) / n,
            'r': pearson([a for a, _ in pairs], [b for _, b in pairs]),
            'mean1': sum(a for a, _ in pairs) / n, 'mean2': sum(b for _, b in pairs) / n}


def build_report3(results, judge2_model):
    L = []
    modes = [m for m in MODES3 if any(m in r['modes'] for r in results)]
    hand = [r for r in results if r['set'] == 'hand']
    synth = [r for r in results if r['set'] == 'synthetic']
    L.append('# Additional Evidence Report (controlled comparison, objective similarity, second judge)')
    L.append('')
    L.append('- Questions: %d (hand-written %d, synthetic %d). Scores 1-5, mean \u00b1 SD.' % (len(results), len(hand), len(synth)))
    L.append('- Main judge: %s | Second judge: %s' % (', '.join(sorted({r.get('judge_model') for r in results if r.get('judge_model')})) or '-', judge2_model or '-'))
    L.append('')
    L.append('## 1. Controlled ablation')
    L.append('')
    L.append('"Hybrid, strict prompt" uses the production hybrid prompt plus the rule "use ONLY the context", '
             'so it differs from vector-only in one thing only: the knowledge-graph facts.')
    L.append('')
    rows = []
    for mode in modes:
        hr, tot = ev.halluc_rate(results, mode)
        rows.append([LABEL3[mode], len(ev.collect(results, mode, 'faithfulness')),
                     ev.pm(ev.collect(results, mode, 'faithfulness')), ev.pm(ev.collect(results, mode, 'relevance')),
                     ev.pm(ev.collect(results, mode, 'correctness')),
                     ('%.1f%% (%d claims)' % (100 * hr, tot)) if hr is not None else '-'])
    L.append(ev.md_table(['System', 'n', 'Faithfulness', 'Relevance', 'Correctness*', 'Hallucination rate**'], rows))
    L.append('')
    L.append('\\* only for the generated questions (they have a reference answer). \\** unsupported claims / all claims.')
    L.append('')
    if hand:
        cats = sorted({r['category'] for r in hand})
        rows = []
        for c in cats:
            rows.append([c] + ['%s / %s' % (ev.fmt(ev.mean(ev.collect(results, m, 'faithfulness', 'hand', c))),
                                            ev.fmt(ev.mean(ev.collect(results, m, 'relevance', 'hand', c)))) for m in modes])
        L.append('**By category, hand-written set (Faithfulness / Relevance)**')
        L.append('')
        L.append(ev.md_table(['Category'] + [LABEL3[m] for m in modes], rows))
        L.append('')

    L.append('## 2. Paired comparisons (difference per question, 95% bootstrap CI)')
    L.append('')
    rows = []
    for a, b in [('hybrid_strict', 'vector'), ('hybrid', 'vector'), ('hybrid_strict', 'hybrid')]:
        if a not in modes or b not in modes:
            continue
        for key, name in [('faithfulness', 'Faithfulness'), ('relevance', 'Relevance'), ('correctness', 'Correctness')]:
            p = ev.paired(results, key, a, b)
            if p:
                sig = 'yes' if (p['lo'] > 0 or p['hi'] < 0) else 'no'
                rows.append(['%s \u2212 %s' % (LABEL3[a], LABEL3[b]), name, p['n'], '%+.2f' % p['mean'],
                             '[%+.2f, %+.2f]' % (p['lo'], p['hi']), '%d / %d / %d' % (p['win'], p['tie'], p['loss']), sig])
    L.append(ev.md_table(['Comparison', 'Metric', 'n', 'Mean diff', '95% CI', 'W / T / L', 'CI excludes 0?'], rows))
    L.append('')
    L.append('A difference is called real only when the CI excludes 0.')
    L.append('')

    L.append('## 3. Objective similarity to the reference answer (generated questions, no LLM judge)')
    L.append('')
    if synth:
        rows = []
        for mode in modes:
            rows.append([LABEL3[mode], len(sim_vals(synth, mode, 'ref_recall')),
                         ev.pm(sim_vals(synth, mode, 'ref_recall'), 3), ev.pm(sim_vals(synth, mode, 'cos'), 3)])
        L.append(ev.md_table(['System', 'n', 'Reference-word recall', 'Embedding cosine'], rows))
        L.append('')
        rows = []
        for a, b in [('hybrid_strict', 'vector'), ('hybrid', 'vector')]:
            for key, name in [('ref_recall', 'Reference-word recall'), ('cos', 'Embedding cosine')]:
                p = paired_sim(synth, key, a, b)
                if p:
                    rows.append(['%s \u2212 %s' % (LABEL3[a], LABEL3[b]), name, p['n'], '%+.3f' % p['mean'],
                                 '[%+.3f, %+.3f]' % (p['lo'], p['hi'])])
        if rows:
            L.append(ev.md_table(['Comparison', 'Metric', 'n', 'Mean diff', '95% CI'], rows))
            L.append('')
        L.append('Reference-word recall = share of the reference answer\'s content words found in the answer. '
                 'Cosine = MiniLM sentence-embedding similarity. Both are rough, objective cross-checks.')
    else:
        L.append('No generated questions in the results.')
    L.append('')

    L.append('## 4. Second-judge agreement (replaces a human check)')
    L.append('')
    rows = []
    for key, name in [('faithfulness', 'Faithfulness'), ('relevance', 'Relevance'), ('correctness', 'Correctness')]:
        a = judge_agreement(results, key)
        if a:
            rows.append([name, a['n'], '%.0f%%' % (100 * a['exact']), '%.0f%%' % (100 * a['within1']),
                         '%.2f' % a['mae'], ev.fmt(a['r']), '%.2f / %.2f' % (a['mean1'], a['mean2'])])
    if rows:
        L.append(ev.md_table(['Metric', 'n scores', 'Exact agreement', 'Within \u00b11 point', 'Mean abs. diff', 'Pearson r', 'Mean (judge 1 / judge 2)'], rows))
        L.append('')
        if judge2_model in sorted({r.get('gen_model') for r in results}):
            L.append('Warning: the second judge is the same model that wrote the answers, so agreement here shows '
                     'consistency, not independence.')
            L.append('')
    else:
        L.append('No second-judge scores yet.')
        L.append('')

    L.append('## 5. How to read this honestly')
    L.append('')
    L.append('- Claims are supported only where the 95% CI excludes 0. Small samples can hide real effects and can also produce lucky ones.')
    L.append('- The generated questions come from one chunk each, so they favour plain vector search; the relational hand-written questions are the fairer test of the graph.')
    L.append('- All judging is automatic. The agreement table shows the two judges are consistent, which is weaker evidence than human agreement.')
    L.append('- Results come from one model family (gpt-oss) and one 18+12 question set on Catalyst 9300 documentation.')
    return '\n'.join(L)


def write_report3(results, judge2_model):
    text = build_report3(results, judge2_model)
    with open(REPORT3, 'w') as f:
        f.write(text + '\n')
    print(text)
    print('\nSaved: %s and %s' % (REPORT3, V3_FILE))


# --------------------------------------------------------------------------- main
def load_start():
    if os.path.exists(V3_FILE):
        with open(V3_FILE) as f:
            return json.load(f)
    if not os.path.exists(ev.RESULTS_FILE):
        print('%s not found. Run evaluation_v2.py first, from this same folder.' % ev.RESULTS_FILE)
        raise SystemExit(1)
    with open(ev.RESULTS_FILE) as f:
        return copy.deepcopy(json.load(f))


def main():
    ap = argparse.ArgumentParser(description='Extra evidence for the Cisco KG-RAG evaluation')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--sample', type=int, default=12, help='questions re-judged by the second judge')
    ap.add_argument('--limit', type=int, default=0, help='only run the first N unfinished strict-hybrid questions')
    args = ap.parse_args()

    results = load_start()
    judge2_model = os.environ.get('JUDGE2_MODEL')
    if not judge2_model:
        for r in results:
            for m in r['modes'].values():
                if m.get('judge2') and m['judge2'].get('model'):
                    judge2_model = m['judge2']['model']
    if args.report:
        write_report3(results, judge2_model)
        return

    if not os.environ.get('GROQ_API_KEY'):
        print('GROQ_API_KEY is not set in this terminal. Run: source ~/.zshrc')
        raise SystemExit(1)
    from groq import Groq
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    if not judge2_model:
        try:
            avail = [m.id for m in client.models.list().data]
        except Exception as e:  # noqa
            avail = []
            print('Could not list models: %s' % str(e)[:100])
        for p in JUDGE2_PREF:
            if p in avail:
                judge2_model = p
                break
        judge2_model = judge2_model or 'openai/gpt-oss-120b'
    print('Second judge model: %s' % judge2_model, flush=True)
    llm = ev.LLM(client)
    llm2 = LLMStrip(client)
    pipe = Pipe3()
    try:
        run_v3(results, pipe, llm, llm2, judge2_model, args.sample, args.limit)
    except ev.FatalError as e:
        print('\nStopped: %s' % str(e)[:300])
        print('Fix the problem above, then run the same command again (progress is saved).')
    except ev.DailyLimit as e:
        print('\nDaily token limit reached. Progress is saved; run the same command again later.')
        print(str(e)[:300])
    save3(results)
    write_report3(results, judge2_model)


if __name__ == '__main__':
    main()
