#!/usr/bin/env python3
"""
Hallucination audit. Makes NO LLM calls, so it uses no Groq tokens.

What it does
  1. Reads eval_v3_results.json (all modes: vector, kg, hybrid, hybrid_strict).
  2. For every answer it rebuilds the context the model saw (retrieved chunks + graph facts).
  3. Splits each answer into sentences and checks each one against that context:
       - word support: share of the sentence's content words found in the context
       - semantic support: highest MiniLM cosine to any context sentence
       - invented specifics: numbers / CLI-like tokens in the sentence that appear nowhere in the context
     A sentence with low word support AND low semantic support, or with an invented specific, is "suspect".
  4. Writes audit_hallucination.html (read it in Safari) and audit_hallucination.csv, and prints a summary.

It is an automatic cross-check of the LLM judge, not ground truth: read the flagged sentences yourself.

Run (from ~/cisco-kg-rag-project):
  python3 -u audit_hallucination.py                 # all modes
  python3 -u audit_hallucination.py --modes vector,hybrid,hybrid_strict --top 20
"""
import argparse, csv, html, json, os, re, sys
from collections import Counter, defaultdict

RESULTS = 'eval_v3_results.json'
OUT_HTML = 'audit_hallucination.html'
OUT_CSV = 'audit_hallucination.csv'

STOP = set('the a an and or of to in on for is are was were be been it its this that with as by at from can '
           'will you your if not no do does which what when how also any all use used using than then there '
           'these those they their have has had may must should would could into only more most such other '
           'each per via'.split())
WORD_T, SEM_T = 0.5, 0.55


def content_tokens(s):
    return [t for t in re.findall(r'[a-z0-9]+', (s or '').lower()) if len(t) > 2 and t not in STOP]


def split_sentences(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    text = re.sub(r'[*#`|]+', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if len(content_tokens(p)) >= 4]


def specifics(s):
    """Numbers and command-like tokens (contain a digit, slash, dash or underscore, or are all caps)."""
    out = set()
    for t in re.findall(r'[A-Za-z0-9_./:-]+', s):
        t = t.strip('.,:;()')
        if len(t) < 2:
            continue
        if re.search(r'\d', t) or re.search(r'[/_]', t) or (t.isupper() and len(t) >= 3):
            out.add(t.lower())
    return out


def is_refusal(s):
    return bool(re.search(r"(not (found|provided|contain|mention|specif|include|cover)|does not contain|"
                          r"no information|cannot (be )?(determine|answer)|doesn't (say|contain))", s.lower()))


def context_sentences(chunks_text, facts):
    sents = []
    for block in chunks_text:
        sents += [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+', block) if len(x.strip()) > 20]
    sents += [f for f in facts if f]
    return sents


def check_answer(answer, ctx_chunks, facts, embed_fn, cosine):
    ctx_sents = context_sentences(ctx_chunks, facts)
    ctx_blob = ' '.join(ctx_chunks) + ' ' + ' '.join(facts)
    ctx_tokens = set(content_tokens(ctx_blob))
    ctx_specs = specifics(ctx_blob)
    fact_tokens = set(content_tokens(' '.join(facts)))
    chunk_tokens = set(content_tokens(' '.join(ctx_chunks)))
    sents = split_sentences(answer)
    if not sents or not ctx_sents:
        return []
    vecs = embed_fn(ctx_sents + sents)
    cv, sv = vecs[:len(ctx_sents)], vecs[len(ctx_sents):]
    rows = []
    for s, v in zip(sents, sv):
        toks = content_tokens(s)
        word = sum(1 for t in toks if t in ctx_tokens) / max(1, len(toks))
        sem = max(cosine(v, c) for c in cv)
        spec_missing = sorted(x for x in specifics(s) if x not in ctx_specs and x not in ctx_blob.lower())
        in_chunks = sum(1 for t in toks if t in chunk_tokens) / max(1, len(toks))
        in_facts = sum(1 for t in toks if t in fact_tokens) / max(1, len(toks))
        refusal = is_refusal(s)
        suspect = (not refusal) and ((word < WORD_T and sem < SEM_T) or bool(spec_missing))
        why = []
        if word < WORD_T and sem < SEM_T:
            why.append('low word+semantic support')
        if spec_missing:
            why.append('specifics not in context: ' + ', '.join(spec_missing[:5]))
        rows.append({'sentence': s, 'word_support': round(word, 2), 'semantic_support': round(sem, 2),
                     'from_chunks': round(in_chunks, 2), 'from_graph': round(in_facts, 2),
                     'refusal': refusal, 'suspect': suspect, 'why': '; '.join(why)})
    return rows


def run(results, modes, chunks_by_ids, embed_fn, cosine, chunk_chars=1200, top_k=5):
    rows = []
    for rec in results:
        ids = (rec.get('retrieved_ids') or [])[:top_k]
        chunks = [c['text'][:chunk_chars] for c in chunks_by_ids(ids)]
        facts = rec.get('graph_facts') or []
        for mode in modes:
            m = (rec.get('modes') or {}).get(mode)
            if not m or not m.get('answer'):
                continue
            if mode == 'vector':
                ctx_chunks, ctx_facts = chunks, []
            elif mode == 'kg':
                ctx_chunks, ctx_facts = [], facts
            else:
                ctx_chunks, ctx_facts = chunks, facts
            j = m.get('judge') or {}
            for r in check_answer(m['answer'], ctx_chunks, ctx_facts, embed_fn, cosine):
                r.update({'id': rec.get('id'), 'mode': mode, 'category': rec.get('category'),
                          'set': rec.get('set'), 'question': rec.get('question'),
                          'judge_unsupported': j.get('unsupported_claims'),
                          'judge_total': j.get('total_claims')})
                rows.append(r)
    return rows


def summarise(rows, results, modes):
    out = []
    for mode in modes:
        rs = [r for r in rows if r['mode'] == mode and not r['refusal']]
        if not rs:
            continue
        sus = sum(1 for r in rs if r['suspect'])
        spec = sum(1 for r in rs if 'specifics' in r['why'])
        per_q = defaultdict(lambda: [0, 0])
        for r in rs:
            per_q[r['id']][0] += 1
            per_q[r['id']][1] += int(r['suspect'])
        jt = ju = 0
        for rec in results:
            j = ((rec.get('modes') or {}).get(mode) or {}).get('judge') or {}
            if j.get('total_claims') is not None and j.get('unsupported_claims') is not None:
                jt += j['total_claims']
                ju += j['unsupported_claims']
        out.append({'mode': mode, 'sentences': len(rs), 'suspect': sus,
                    'suspect_pct': 100.0 * sus / len(rs), 'invented_specific': spec,
                    'judge_unsupported_pct': (100.0 * ju / jt) if jt else None, 'judge_claims': jt})
    return out


def write_html(rows, summary, top):
    css = ("body{font-family:-apple-system,Helvetica,Arial;max-width:1000px;margin:32px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;margin:12px 0;width:100%}th,td{border:1px solid #ccc;padding:6px 10px;"
           "text-align:left;vertical-align:top}th{background:#f0f0f0}.s{background:#fff3cd}")
    h = ['<html><head><meta charset="utf-8"><style>%s</style></head><body>' % css,
         '<h1>Hallucination audit (automatic, no LLM)</h1>',
         '<p>A sentence is <b>suspect</b> when its words and meaning are both weakly supported by the context the model saw, '
         'or when it contains numbers/commands that appear nowhere in that context. This cross-checks the LLM judge; '
         'read the flagged sentences yourself before drawing conclusions.</p>',
         '<h2>Summary</h2><table><tr><th>Mode</th><th>Answer sentences</th><th>Suspect</th><th>Suspect %</th>'
         '<th>With invented specifics</th><th>Judge: unsupported claims %</th></tr>']
    for s in summary:
        jp = '-' if s['judge_unsupported_pct'] is None else '%.1f%% (%d claims)' % (s['judge_unsupported_pct'], s['judge_claims'])
        h.append('<tr><td>%s</td><td>%d</td><td>%d</td><td>%.1f%%</td><td>%d</td><td>%s</td></tr>' % (
            s['mode'], s['sentences'], s['suspect'], s['suspect_pct'], s['invented_specific'], jp))
    h.append('</table>')
    for s in summary:
        sus = [r for r in rows if r['mode'] == s['mode'] and r['suspect']]
        sus.sort(key=lambda r: (r['word_support'] + r['semantic_support']))
        h.append('<h2>%s: most suspect sentences (top %d of %d)</h2><table><tr><th>Question</th><th>Sentence</th>'
                 '<th>Word</th><th>Meaning</th><th>Why flagged</th></tr>' % (s['mode'], min(top, len(sus)), len(sus)))
        for r in sus[:top]:
            h.append('<tr class="s"><td>%s<br><small>%s / %s</small></td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%s</td></tr>' % (
                html.escape(str(r['question'])[:140]), html.escape(str(r['set'])), html.escape(str(r['category'])),
                html.escape(r['sentence']), r['word_support'], r['semantic_support'], html.escape(r['why'])))
        h.append('</table>')
    h.append('</body></html>')
    open(OUT_HTML, 'w', encoding='utf-8').write('\n'.join(h))


def write_csv(rows):
    cols = ['id', 'mode', 'set', 'category', 'question', 'sentence', 'word_support', 'semantic_support',
            'from_chunks', 'from_graph', 'refusal', 'suspect', 'why', 'judge_unsupported', 'judge_total']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=RESULTS)
    ap.add_argument('--modes', default='vector,kg,hybrid,hybrid_strict')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--no-open', action='store_true')
    a = ap.parse_args()
    modes = [m for m in a.modes.split(',') if m]
    results = json.load(open(a.results))
    import evaluation_v3 as v3
    pipe = v3.Pipe3()

    def embed_fn(texts):
        return pipe.embed.encode(texts, batch_size=64).tolist()

    rows = run(results, modes, pipe.chunks_by_ids, embed_fn, v3.cosine)
    summary = summarise(rows, results, modes)
    write_csv(rows)
    write_html(rows, summary, a.top)
    print('\nMode              sentences  suspect  suspect%  invented-specifics  judge-unsupported%')
    for s in summary:
        jp = '-' if s['judge_unsupported_pct'] is None else '%.1f%%' % s['judge_unsupported_pct']
        print('%-16s  %9d  %7d  %7.1f%%  %18d  %18s' % (s['mode'], s['sentences'], s['suspect'],
                                                         s['suspect_pct'], s['invented_specific'], jp))
    print('\nWrote %s and %s' % (OUT_HTML, OUT_CSV))
    if not a.no_open and sys.platform == 'darwin':
        os.system('open -a Safari "%s"' % os.path.abspath(OUT_HTML))


if __name__ == '__main__':
    main()
