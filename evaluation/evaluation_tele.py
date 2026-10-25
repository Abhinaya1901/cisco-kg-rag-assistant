#!/usr/bin/env python3
"""
evaluation_tele.py -- the SAME automated evaluation as the Cisco project, run on the TeleQnA
3GPP Release 18 subset.  It re-uses evaluation_v2.py and evaluation_v3.py unchanged (metrics, judge,
statistics, report) and only swaps the data: corpus, questions, and the graph.

Run from the folder that contains evaluation_v2.py, evaluation_v3.py, chroma_3gpp/, 3gpp_chunks.json,
tele_r18_questions.json and tele_r18_coverage.json (the teleqna folder, with the two evaluation files copied in).

  python3 -u evaluation_tele.py --limit 1          # smoke test: 1 question
  python3 -u evaluation_tele.py --n 60             # main run (resumable), writes evaluation_report_tele.md
  python3 -u evaluation_tele.py --report           # rebuild the report only
  python3 -u evaluation_tele.py v3 --sample 12     # controlled strict run + similarity + 2nd judge (resumable)
  python3 -u evaluation_tele.py v3 --report

Differences from the Cisco run, stated openly:
  * Questions: TeleQnA multiple-choice items turned into open questions (options removed); the reference
    answer is the dataset's own explanation plus the correct option text.
  * Only questions that the corpus can answer are used (word overlap with the explanation >= 0.6).
  * Gold chunk = the corpus chunk with the highest word overlap with the explanation (objective, no LLM).
  * Graph = a document-structure graph (cross-references between clauses/specs + neighbouring passages),
    built without an LLM, because LLM triple extraction over 25k chunks does not fit the free token limits.
"""
import os
import re
import sys
import json
import random

import evaluation_v2 as ev
import evaluation_v3 as v3

CHUNKS_FILE = '3gpp_chunks.json'
QUESTIONS_FILE = 'tele_r18_questions.json'
COVERAGE_FILE = 'tele_r18_coverage.json'
ITEMS_FILE = 'tele_items.json'
OVERLAP_MIN = 0.6
SNIPPET_CHARS = 350
MAX_FACTS = 6
TELE_MODES = ['vector', 'hybrid']

STOP = set("""that this with from which shall will have been their there these those when where while into
than then also such only used using must should would could does done each other more most some being both
between within without under over after before about they them what whose were not and the for are can may
its has had any all one two""".split())

CLAUSE_RE = re.compile(r'^\s*([A-Z]?[\.\d]*\d)')
XREF_SPEC_RE = re.compile(r'clauses?\s+([A-Z]?\d+(?:\.\d+)*)\s+of\s+(?:3GPP\s+)?TS\s+(\d\d\.\d{3})')
XREF_SAME_RE = re.compile(r'\bclauses?\s+([A-Z]?\d+(?:\.\d+)*)(?!\s+of\s+(?:3GPP\s+)?TS)')
PREFIX_RE = re.compile(r'^\[[^\]]*\]\s*')
CLOSED_STYLE_RE = re.compile(r'\b(following|except|none of the above|which statement|which of these)\b', re.I)


def toks(s):
    return {t for t in re.findall(r'[a-z0-9]+', (s or '').lower()) if len(t) > 3 and t not in STOP}


def overlap(exp, text):
    return len(exp & toks(text)) / max(1, len(exp))


def clause_of(section):
    m = CLAUSE_RE.match(section or '')
    return m.group(1) if m else None


# --------------------------------------------------------------------------- pipeline
class TelePipeline(ev.Pipeline):
    """Same interface as the Cisco Pipeline (retrieve / graph_facts / all_chunks / chunks_by_ids)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb
        print('Loading embedding model, 3GPP index and document graph...', flush=True)
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')
        self.col = chromadb.PersistentClient(path='./chroma_3gpp').get_collection('tele_3gpp')
        with open(CHUNKS_FILE) as f:
            self.chunks = json.load(f)
        self._build_graph()
        self._last_ids = []
        print('Loaded. %d chunks, %d cross-reference edges, %d continuation edges'
              % (len(self.chunks), self.n_xref, self.n_next), flush=True)

    # -- graph ---------------------------------------------------------------
    def _build_graph(self):
        self.by_id = {c['id']: i for i, c in enumerate(self.chunks)}
        first = {}
        for c in self.chunks:
            cl = clause_of(c.get('section'))
            if cl:
                first.setdefault((c['spec'], cl), c['id'])
        self.xref = {}
        self.nxt = {}
        self.n_xref = self.n_next = 0
        for i, c in enumerate(self.chunks):
            targets = []
            text = c['text']
            for cl, spec in XREF_SPEC_RE.findall(text):
                targets.append((spec, cl))
            for cl in XREF_SAME_RE.findall(text):
                targets.append((c['spec'], cl))
            seen, out = set(), []
            for key in targets:
                tid = first.get(key)
                if tid and tid != c['id'] and tid not in seen:
                    seen.add(tid)
                    out.append(tid)
            if out:
                self.xref[c['id']] = out[:3]
                self.n_xref += len(self.xref[c['id']])
            if i + 1 < len(self.chunks):
                n = self.chunks[i + 1]
                if n['spec'] == c['spec'] and n.get('section') == c.get('section'):
                    self.nxt[c['id']] = n['id']
                    self.n_next += 1

    def _label(self, cid):
        c = self.chunks[self.by_id[cid]]
        cl = clause_of(c.get('section'))
        return 'TS %s %s' % (c['spec'], ('clause ' + cl) if cl else (c.get('section') or '')[:40])

    def _snippet(self, cid):
        t = PREFIX_RE.sub('', self.chunks[self.by_id[cid]]['text'])
        return t[:SNIPPET_CHARS].replace('\n', ' ')

    # -- interface used by evaluation_v2 -------------------------------------
    def retrieve(self, q, k):
        emb = self.embed.encode([q], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=k)
        ids, docs = res['ids'][0], res['documents'][0]
        metas = (res.get('metadatas') or [[None] * len(ids)])[0]
        self._last_ids = list(ids)
        return [{'id': i, 'text': d, 'meta': m} for i, d, m in zip(ids, docs, metas)]

    def all_chunks(self):
        return [{'id': c['id'], 'text': c['text'], 'meta': {'spec': c['spec'], 'section': c.get('section')}}
                for c in self.chunks]

    def chunks_by_ids(self, ids):
        res = self.col.get(ids=list(ids), include=['documents'])
        m = dict(zip(res['ids'], res['documents']))
        return [{'id': i, 'text': m.get(i, ''), 'meta': None} for i in ids if i in m]

    def graph_facts(self, entities, max_facts=MAX_FACTS):
        """Graph expansion around the top retrieved chunks: cross-referenced clauses first, then the next
        passage of the same section. Passages already in the generator's top-5 are skipped."""
        shown = set(self._last_ids[:ev.TOP_K_GEN])
        facts, used = [], set()
        for seed in self._last_ids[:3]:
            if seed not in self.by_id:
                continue
            cand = [('cross_reference', t) for t in self.xref.get(seed, [])]
            if seed in self.nxt:
                cand.append(('continues_with', self.nxt[seed]))
            for rel, tid in cand:
                if tid in shown or tid in used:
                    continue
                used.add(tid)
                facts.append('%s --%s--> %s: "%s"' % (self._label(seed), rel, self._label(tid), self._snippet(tid)))
                if len(facts) >= max_facts:
                    return facts
        return facts


# --------------------------------------------------------------------------- items
def build_items(pipe, n):
    """Open-ended questions with objective gold chunks. Deterministic (seed 42), cached in tele_items.json."""
    items = []
    if os.path.exists(ITEMS_FILE):
        with open(ITEMS_FILE) as f:
            items = json.load(f)
        if len(items) >= n:
            return items[:n]
    qs = {q['id']: q for q in json.load(open(QUESTIONS_FILE))}
    cov = [r for r in json.load(open(COVERAGE_FILE)) if r.get('scorable') and r['oracle'] >= OVERLAP_MIN]
    cand = sorted(r['id'] for r in cov if not CLOSED_STYLE_RE.search(qs[r['id']]['question']))
    random.Random(42).shuffle(cand)
    print('Candidate questions after filters: %d' % len(cand), flush=True)
    items = []
    for qid in cand:
        if len(items) >= max(n, 200):
            break
        q = qs[qid]
        exp = toks(q['explanation'])
        answer = q['options'][q['answer_idx']]
        emb = pipe.embed.encode([q['question'] + ' ' + answer], normalize_embeddings=True).tolist()
        res = pipe.col.query(query_embeddings=emb, n_results=10, include=['documents'])
        scored = [(overlap(exp, d), i) for i, d in zip(res['ids'][0], res['documents'][0])]
        best, gold = max(scored)
        if best < OVERLAP_MIN:
            continue
        ref = q['explanation'].strip()
        if answer.lower() not in ref.lower():
            ref = '%s (Correct answer: %s)' % (ref, answer)
        items.append({'id': 'T%03d' % (len(items) + 1), 'set': 'synthetic', 'category': 'teleqna-r18',
                      'question': q['question'], 'reference': ref, 'gold_chunk_id': gold,
                      'source_id': qid, 'gold_overlap': round(best, 3)})
    with open(ITEMS_FILE, 'w') as f:
        json.dump(items, f, indent=1)
    print('Built %d open-ended items -> %s' % (len(items), ITEMS_FILE), flush=True)
    return items[:n]


# --------------------------------------------------------------------------- wiring
def patch():
    ev.RESULTS_FILE = 'eval_tele_results.json'
    ev.REPORT_FILE = 'evaluation_report_tele.md'
    ev.CSV_FILE = 'eval_tele_per_question.csv'
    ev.MODES_ALL = list(TELE_MODES)
    ev.Pipeline = TelePipeline
    ev.extract_entities = lambda llm, model, question: []        # graph needs no LLM call here
    ev.write_audit = lambda results, *a, **k: False              # no human audit file
    v3.V3_FILE = 'eval_tele_v3_results.json'
    v3.REPORT3 = 'evaluation_report_tele_v3.md'
    v3.MODES3 = ['vector', 'hybrid', 'hybrid_strict']
    v3.Pipe3 = TelePipeline

    orig = ev.build_report
    header = ('# Automated Evaluation Report: TeleQnA (3GPP Release 18 subset)\n\n'
              '- Corpus: 25 3GPP Release 18 specifications. Questions: TeleQnA "Standards specifications" items '
              'tagged Release 18, converted to open questions (options removed) and kept only if the corpus covers '
              'them (word overlap with the dataset explanation >= %.1f).\n'
              '- Reference answer = the dataset explanation (+ the correct option). Gold chunk = the chunk with the '
              'highest word overlap with the explanation (objective, no LLM).\n'
              '- "Hybrid" here = vector top-5 plus a document-structure graph (cross-references and continuation '
              'passages), not an LLM-extracted entity graph.\n' % OVERLAP_MIN)

    def build_report(results):
        t = orig(results)
        t = t.replace('# Automated Evaluation Report', header, 1)
        t = t.replace('- Judging is done by an LLM. See judge_audit.csv: a human spot-check of a sample, summarised by `--agreement`.',
                      '- Judging is done by an LLM; a second, different judge model re-scores a sample in the v3 stage (no human scoring).')
        return t
    ev.build_report = build_report


def main():
    import argparse
    ap = argparse.ArgumentParser(description='TeleQnA evaluation (re-uses the Cisco evaluation code)')
    ap.add_argument('--n', type=int, default=60, help='number of questions (extend later; progress is kept)')
    ap.add_argument('--modes', default=','.join(TELE_MODES))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    results = ev.load_results()
    if args.report:
        ev.write_report(results)
        return
    if not os.environ.get('GROQ_API_KEY'):
        print('GROQ_API_KEY is not set in this terminal. Run: source ~/.zshrc')
        raise SystemExit(1)
    from groq import Groq
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    gen_model = ev.pick_model(client, ev.GEN_MODEL_PREF, os.environ.get('GEN_MODEL'), 'generation')
    judge_model = ev.pick_model(client, ev.JUDGE_MODEL_PREF, os.environ.get('JUDGE_MODEL'), 'judging')
    print('Generation model: %s | Judge model: %s' % (gen_model, judge_model), flush=True)
    llm = ev.LLM(client)
    pipe = ev.Pipeline()
    items = build_items(pipe, args.n)
    modes = [m for m in args.modes.split(',') if m in TELE_MODES]
    try:
        ev.run_all(items, pipe, llm, gen_model, judge_model, modes, results, args.limit, args.retry_failed)
    except ev.FatalError as e:
        print('\nStopped: %s' % str(e)[:300])
    ev.write_report(results)


if __name__ == '__main__':
    patch()
    if len(sys.argv) > 1 and sys.argv[1] == 'v3':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        v3.main()
    else:
        main()
