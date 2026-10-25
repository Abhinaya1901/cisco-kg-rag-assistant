#!/usr/bin/env python3
"""
evaluation_v2.py -- automated evaluation of the Cisco KG-RAG assistant.

Run from the project root (the folder that contains chroma_db/ and knowledge_graph.gml).

  python3 -u evaluation_v2.py --only hand --limit 1     # smoke test, 1 question
  python3 -u evaluation_v2.py --only hand               # the 18 hand-written questions
  python3 -u evaluation_v2.py --only synthetic          # auto-generated questions with gold chunk + reference answer
  python3 -u evaluation_v2.py --report                  # rebuild the report from saved results
  python3 -u evaluation_v2.py --agreement               # judge-vs-human agreement after you fill judge_audit.csv

What it measures (all automatic):
  Retrieval : Precision@5, Recall@5, MRR, nDCG@5, Hit@5, Hit@10
              - hand questions: LLM-judged passage relevance (one batched call per question)
              - synthetic questions: objective, the gold source chunk must appear in the top-k
  Answers   : Faithfulness, Relevance, Correctness (vs reference), claim-level Hallucination rate
              for three systems: vector-only, KG-only, hybrid (ablation)
  Stats     : paired bootstrap CI + win/tie/loss for hybrid vs vector-only
  Efficiency: retrieval / KG traversal / generation latency, tokens per answer, KG coverage
Results are saved after every question, so you can stop and resume at any time.
"""
import os
import sys
import re
import json
import time
import math
import random
import csv
import argparse
import statistics
from collections import deque, defaultdict

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

RESULTS_FILE = 'eval_v2_results.json'
SYNTH_FILE = 'synthetic_questions.json'
REPORT_FILE = 'evaluation_report.md'
CSV_FILE = 'eval_v2_per_question.csv'
AUDIT_FILE = 'judge_audit.csv'

TPM_LIMIT = int(os.environ.get('TPM_LIMIT', '6000'))      # stay under Groq's 8000 tokens/min
CHUNK_CHARS = int(os.environ.get('CHUNK_CHARS', '1000'))   # generator AND judge see the same text
TOP_K_GEN = 5
TOP_K_RET = 10
GEN_MAX_TOKENS = 900
JUDGE_MAX_TOKENS = 600

GEN_MODEL_PREF = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']
JUDGE_MODEL_PREF = ['openai/gpt-oss-20b', 'openai/gpt-oss-120b']

MODES_ALL = ['vector', 'kg', 'hybrid']
MODE_LABEL = {'vector': 'Vector-only', 'kg': 'KG-only', 'hybrid': 'Hybrid (KG+Vector)'}

# (id, category, question). Categories are an assumption, edit freely.
HAND_QUESTIONS = [
    ("H01", "factual", "What command enables privileged EXEC mode?"),
    ("H02", "factual", "What is the default VTP mode?"),
    ("H03", "factual", "How many access ports can a Catalyst 9300 stack support?"),
    ("H04", "factual", "What command sets an interface to trunk mode?"),
    ("H05", "factual", "What port speeds does the Catalyst 9300 support?"),
    ("H06", "factual", "What command enters global configuration mode?"),
    ("H07", "relational", "What do I need to configure before setting up a trunk port?"),
    ("H08", "relational", "What does VLAN configuration depend on?"),
    ("H09", "relational", "If I want to use many VLANs without enabling routing, what feature should I configure and why?"),
    ("H10", "relational", "What must I check before upgrading the ROMMON on a switch stack?"),
    ("H11", "relational", "What's required before Spanning Tree Protocol can use PVST mode?"),
    ("H12", "relational", "What are the prerequisites for configuring EtherChannel between two switches?"),
    ("H13", "relational", "If I enable 802.1x on a port, what happens if I try to change it to trunk mode?"),
    ("H14", "relational", "What feature must be enabled to support more than the default number of VLANs?"),
    ("H15", "troubleshooting", "My switch keeps reloading unexpectedly in a stack — what should I check, and what's the relevant command?"),
    ("H16", "troubleshooting", "What known issues exist in the current release notes related to VLAN or trunking?"),
    ("H17", "troubleshooting", "Which command would I use to verify VLAN configuration, and what does the configuration guide say I should check first?"),
    ("H18", "troubleshooting", "What hardware limitation from the datasheet should I be aware of when configuring stacking?"),
]


# --------------------------------------------------------------------------- errors / helpers
class FatalError(Exception):
    pass


class DailyLimit(Exception):
    pass


def parse_wait(msg):
    """'Please try again in 7m26.4s' / '850ms' / '1h2m3s' -> seconds (or None)."""
    m = re.search(r'try again in\s+([0-9hms.]+)', msg)
    if not m:
        return None
    total = 0.0
    found = False
    for val, unit in re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)', m.group(1)):
        v = float(val)
        total += {'ms': v / 1000.0, 'h': v * 3600.0, 'm': v * 60.0, 's': v}[unit]
        found = True
    return total if found else None


class LLM:
    """Groq wrapper: per-model tokens-per-minute throttle, retries, usage accounting."""

    def __init__(self, client, tpm_limit=TPM_LIMIT, sleep=time.sleep, clock=time.time):
        self.client = client
        self.limit = tpm_limit
        self.win = defaultdict(deque)
        self.use_effort = True
        self.tokens = defaultdict(int)
        self.calls = defaultdict(int)
        self._sleep = sleep
        self._clock = clock

    def _throttle(self, model, est):
        dq = self.win[model]
        while True:
            now = self._clock()
            while dq and now - dq[0][0] >= 60:
                dq.popleft()
            used = sum(t for _, t in dq)
            if not dq or used + est <= self.limit:
                return
            self._sleep(max(60 - (now - dq[0][0]) + 0.5, 0.5))

    def call(self, model, prompt, max_tokens=JUDGE_MAX_TOKENS, retries=6):
        est = len(prompt) // 3 + max_tokens
        last_err = None
        attempt = 0
        rate_waits = 0
        while attempt < retries:
            attempt += 1
            self._throttle(model, est)
            t0 = self._clock()
            try:
                kwargs = dict(model=model,
                              messages=[{'role': 'user', 'content': prompt}],
                              temperature=0,
                              max_completion_tokens=max_tokens)
                if self.use_effort:
                    kwargs['reasoning_effort'] = 'low'
                r = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa
                msg = str(e)
                low = msg.lower()
                last_err = msg
                if self.use_effort and 'reasoning_effort' in low:
                    self.use_effort = False
                    attempt -= 1
                    continue
                if 'error code: 401' in low or 'invalid_api_key' in low or 'model_not_found' in low or 'error code: 404' in low:
                    raise FatalError(msg)
                if 'error code: 429' in low or 'rate_limit' in low or 'rate limit' in low:
                    wait = parse_wait(msg)
                    if ('per day' in low or 'tpd' in low) and (wait is None or wait > 900):
                        raise DailyLimit(msg)
                    rate_waits += 1
                    if rate_waits > 25:
                        break
                    w = (wait if wait is not None else 30.0) + 2.0
                    print('    [rate limited on %s, waiting %.0fs]' % (model, w), flush=True)
                    self._sleep(w)
                    attempt -= 1
                    continue
                if 'connection' in low or 'timed out' in low or 'timeout' in low or 'error code: 5' in low:
                    print('    [network problem, retry %d/%d]' % (attempt, retries), flush=True)
                    self._sleep(5 * attempt)
                    continue
                print('    [error: %s]' % msg[:200], flush=True)
                self._sleep(2)
                continue
            ch = r.choices[0]
            text = (ch.message.content or '').strip()
            usage = getattr(r, 'usage', None)
            toks = getattr(usage, 'total_tokens', None) or est
            self.win[model].append((self._clock(), toks))
            self.tokens[model] += toks
            self.calls[model] += 1
            if not text:
                fr = getattr(ch, 'finish_reason', None)
                last_err = 'empty content (finish_reason=%s)' % fr
                if fr == 'length':
                    max_tokens *= 2
                    est = len(prompt) // 3 + max_tokens
                continue
            return {'text': text, 'tokens': toks, 'latency': self._clock() - t0, 'error': None}
        return {'text': None, 'tokens': 0, 'latency': 0.0, 'error': last_err}


def pick_model(client, prefs, override, label):
    if override:
        return override
    try:
        avail = [m.id for m in client.models.list().data]
    except Exception as e:  # noqa
        print('Could not list models (%s); using %s' % (str(e)[:80], prefs[0]))
        return prefs[0]
    for p in prefs:
        if p in avail:
            return p
    print('None of %s are available on your account for %s.' % (prefs, label))
    print('Available models: %s' % ', '.join(sorted(avail)))
    print('Set one yourself, e.g.:  GEN_MODEL=<id> JUDGE_MODEL=<id> python3 -u evaluation_v2.py ...')
    raise SystemExit(1)


# --------------------------------------------------------------------------- retrieval metrics
def precision_at_k(rel, k):
    top = rel[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(rel, k, total_relevant):
    if total_relevant == 0:
        return None
    return sum(rel[:k]) / total_relevant


def reciprocal_rank(rel):
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(rel, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))


def ndcg_at_k(rel, k):
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0


def hit_rate_at_k(rel, k):
    return 1.0 if any(rel[:k]) else 0.0


def compute_retrieval(rel, gold=False):
    total = sum(rel)
    m = {
        'precision_at_5': precision_at_k(rel, 5),
        'recall_at_5': recall_at_k(rel, 5, total),
        'mrr': reciprocal_rank(rel),
        'ndcg_at_5': ndcg_at_k(rel, 5),
        'hit_rate_at_5': hit_rate_at_k(rel, 5),
        'hit_rate_at_10': hit_rate_at_k(rel, 10),
        'total_relevant_in_top10': total,
    }
    if gold:
        m['recall_at_5'] = hit_rate_at_k(rel, 5)
    return m


# --------------------------------------------------------------------------- real pipeline
class Pipeline:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb
        import networkx as nx
        print('Loading embedding model, ChromaDB and knowledge graph...', flush=True)
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')
        self.col = chromadb.PersistentClient(path='./chroma_db').get_collection('cisco_docs')
        self.G = nx.read_gml('knowledge_graph.gml')
        print('Loaded. Graph: %d nodes, %d edges' % (self.G.number_of_nodes(), self.G.number_of_edges()), flush=True)

    def retrieve(self, q, k):
        emb = self.embed.encode([q]).tolist()
        res = self.col.query(query_embeddings=emb, n_results=k)
        ids = res['ids'][0]
        docs = res['documents'][0]
        metas = (res.get('metadatas') or [[None] * len(ids)])[0]
        return [{'id': i, 'text': d, 'meta': m} for i, d, m in zip(ids, docs, metas)]

    def all_chunks(self):
        r = self.col.get(include=['documents', 'metadatas'])
        return [{'id': i, 'text': d, 'meta': m} for i, d, m in zip(r['ids'], r['documents'], r['metadatas'])]

    def _match_nodes(self, entity, max_matches=2):
        entity = entity.lower().strip()
        out = []
        for node in self.G.nodes():
            n = str(node)
            if entity in n or n in entity:
                out.append(node)
            if len(out) >= max_matches:
                break
        return out

    def graph_facts(self, entities, max_facts=8):
        facts = []
        for entity in entities:
            for node in self._match_nodes(entity):
                for nb in list(self.G.neighbors(node))[:3]:
                    ed = self.G.get_edge_data(node, nb) or {}
                    facts.append('%s --%s--> %s' % (node, ed.get('relation', 'related_to'), nb))
                if len(facts) >= max_facts:
                    return facts
        return facts


# --------------------------------------------------------------------------- prompts and judges
def chunks_text(chunks):
    return '\n\n'.join(c['text'][:CHUNK_CHARS] for c in chunks[:TOP_K_GEN])


def facts_text(facts):
    return '\n'.join(facts) if facts else 'None'


def build_prompt_and_context(mode, q, chunks, facts):
    ct, ft = chunks_text(chunks), facts_text(facts)
    if mode == 'vector':
        ctx = ct
        prompt = ('You are a network engineering assistant. Answer the question using ONLY the context below.\n\n'
                  'Context:\n%s\n\nQuestion: %s\n\nAnswer:' % (ctx, q))
    elif mode == 'kg':
        ctx = ft
        prompt = ('You are a network engineering assistant. Answer the question using ONLY the knowledge graph '
                  'relationships below. If they are not enough to answer, say that the knowledge graph does not '
                  'contain enough information.\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n%s\n\nQuestion: %s\n\nAnswer:' % (ctx, q))
    else:
        ctx = '%s\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n%s' % (ct, ft)
        prompt = ('You are a network engineering assistant. Answer the question using the context below.\n\n'
                  'DOCUMENT EXCERPTS:\n%s\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n%s\n\nQuestion: %s\n\n'
                  'Answer in plain conversational paragraphs. Do NOT use markdown tables, headers, or bullet '
                  'checklists.' % (ct, ft, q))
    return prompt, ctx


def extract_entities(llm, model, question):
    out = llm.call(model, '''Extract key technical entities (features, protocols, commands, interfaces) from this question. Return ONLY a JSON list of lowercase strings.

Question: %s

JSON list:''' % question, max_tokens=300)
    if not out['text']:
        return []
    m = re.search(r'\[.*\]', out['text'], re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).lower().strip() for x in arr if str(x).strip()]
    except ValueError:
        return []


def judge_chunk_relevance(llm, model, question, chunks):
    passages = '\n\n'.join('[%d] %s' % (i + 1, c['text'][:500]) for i, c in enumerate(chunks))
    n = len(chunks)
    prompt = '''You are a strict evaluator of document retrieval for a network-engineering QA system.

Question: %s

Below are %d retrieved passages. For each passage decide whether it contains information that directly helps answer the question (1) or not (0).

%s

Respond with ONLY a JSON list of %d integers (each 0 or 1), in passage order. Example: [1,0,0,1,0]''' % (question, n, passages, n)
    out = None
    for _ in range(2):
        out = llm.call(model, prompt, max_tokens=JUDGE_MAX_TOKENS)
        if out['text']:
            for m in re.finditer(r'\[[\s01,]*\]', out['text']):
                try:
                    arr = json.loads(m.group(0))
                except ValueError:
                    continue
                if len(arr) == n:
                    return [int(x) for x in arr], out
    return None, out


def _grab_int(pattern, text, lo, hi):
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    v = int(m.group(1))
    return v if lo <= v <= hi else None


def judge_answer(llm, model, question, context, answer, reference=None):
    ref_block = ''
    corr_rule = '4. CORRECTNESS: write NA (no reference answer is available).'
    corr_line = 'CORRECTNESS: NA'
    if reference:
        ref_block = '\nReference answer (ground truth written from the source document):\n"""%s"""\n' % reference
        corr_rule = ('4. CORRECTNESS (1-5): compare the answer with the reference. 5 = agrees on all key facts; '
                     '3 = partly agrees or misses a key fact; 1 = contradicts the reference or misses everything.')
        corr_line = 'CORRECTNESS: <1-5>'
    prompt = '''You are a strict evaluator of a retrieval-augmented question answering system.

Question: %s

Context the assistant was given:
"""%s"""
%s
Assistant's answer:
"""%s"""

Evaluate the answer:
1. Split the answer into its individual factual claims. TOTAL_CLAIMS = how many claims. UNSUPPORTED_CLAIMS = how many of them are NOT supported by the context above (facts that are only from general knowledge and not in the context count as unsupported).
2. FAITHFULNESS (1-5): 5 = every claim is supported by the context; 3 = some claims are unsupported; 1 = mostly unsupported.
3. RELEVANCE (1-5): 5 = directly and completely answers the question; 3 = partially; 1 = off-topic or a non-answer.
%s

Respond in EXACTLY this format, nothing else:
TOTAL_CLAIMS: <integer>
UNSUPPORTED_CLAIMS: <integer>
FAITHFULNESS: <1-5>
RELEVANCE: <1-5>
%s''' % (question, context, ref_block, answer, corr_rule, corr_line)
    for _ in range(2):
        out = llm.call(model, prompt, max_tokens=JUDGE_MAX_TOKENS)
        if not out['text']:
            continue
        t = out['text']
        faith = _grab_int(r'FAITHFULNESS:\s*(\d+)', t, 1, 5)
        rel = _grab_int(r'RELEVANCE:\s*(\d+)', t, 1, 5)
        if faith is None or rel is None:
            continue
        total = _grab_int(r'TOTAL_CLAIMS:\s*(\d+)', t, 0, 200)
        unsup = _grab_int(r'UNSUPPORTED_CLAIMS:\s*(\d+)', t, 0, 200)
        if total is not None and unsup is not None:
            unsup = min(unsup, total)
        corr = _grab_int(r'CORRECTNESS:\s*(\d+)', t, 1, 5) if reference else None
        return {'faithfulness': faith, 'relevance': rel, 'correctness': corr,
                'total_claims': total, 'unsupported_claims': unsup, 'judge_tokens': out['tokens']}
    return None


# --------------------------------------------------------------------------- synthetic question set
def build_synthetic(pipe, llm, model, n, seed=42):
    data = []
    if os.path.exists(SYNTH_FILE):
        with open(SYNTH_FILE) as f:
            data = json.load(f)
    if len(data) >= n:
        return data[:n]
    used = {d['gold_chunk_id'] for d in data}
    pool = []
    for c in pipe.all_chunks():
        t = c['text'] or ''
        if 400 <= len(t) <= 2500 and len(t.split()) >= 60:
            ok = sum(ch.isalnum() or ch.isspace() for ch in t) / len(t)
            if ok > 0.80 and c['id'] not in used:
                pool.append(c)
    random.Random(seed).shuffle(pool)
    print('Generating synthetic questions from random chunks (pool of %d)...' % len(pool), flush=True)
    tried = 0
    for c in pool:
        if len(data) >= n or tried >= n * 5:
            break
        tried += 1
        text = c['text'][:CHUNK_CHARS]
        prompt = '''You are helping build a test set for a Cisco Catalyst 9300 documentation assistant.

Passage:
"""%s"""

Write ONE specific, self-contained question that a network engineer might ask and that this passage fully answers. Do not mention "the passage", "the document" or "the text" in the question. Then write the reference answer in 1-3 sentences, using ONLY facts stated in the passage.

Respond with ONLY a JSON object: {"question": "...", "reference": "..."}''' % text
        out = llm.call(model, prompt, max_tokens=700)
        if not out['text']:
            continue
        m = re.search(r'\{.*\}', out['text'], re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0), strict=False)
        except ValueError:
            continue
        q, ref = str(obj.get('question', '')).strip(), str(obj.get('reference', '')).strip()
        if len(q) < 15 or len(ref) < 15 or re.search(r'\b(passage|document|the text)\b', q, re.I):
            continue
        data.append({'id': 'S%02d' % (len(data) + 1), 'set': 'synthetic', 'category': 'synthetic',
                     'question': q, 'reference': ref, 'gold_chunk_id': c['id'], 'meta': c.get('meta')})
        with open(SYNTH_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print('  %s: %s' % (data[-1]['id'], q[:90]), flush=True)
    return data[:n]


# --------------------------------------------------------------------------- one question
def evaluate_question(item, pipe, llm, gen_model, judge_model, modes):
    q = item['question']
    rec = {'id': item['id'], 'set': item['set'], 'category': item['category'], 'question': q,
           'reference': item.get('reference'), 'gold_chunk_id': item.get('gold_chunk_id'),
           'gen_model': gen_model, 'judge_model': judge_model, 'modes': {}, 'failed': False}

    t0 = time.time()
    chunks = pipe.retrieve(q, TOP_K_RET)
    rec['timing'] = {'retrieval_ms': (time.time() - t0) * 1000.0}
    rec['retrieved_ids'] = [c['id'] for c in chunks]

    # retrieval quality
    if item.get('gold_chunk_id'):
        rel = [1 if c['id'] == item['gold_chunk_id'] else 0 for c in chunks]
        rec['retrieval'] = compute_retrieval(rel, gold=True)
    else:
        rel, _ = judge_chunk_relevance(llm, judge_model, q, chunks)
        rec['retrieval'] = compute_retrieval(rel) if rel is not None else None
        rec['relevance_list'] = rel
        if rel is None:
            rec['failed'] = True

    # knowledge graph side
    entities = extract_entities(llm, gen_model, q)
    t1 = time.time()
    facts = pipe.graph_facts(entities)
    rec['timing']['kg_traversal_ms'] = (time.time() - t1) * 1000.0
    rec['entities'] = entities
    rec['graph_facts'] = facts
    rec['num_graph_facts'] = len(facts)

    # answers + judging
    for mode in modes:
        prompt, ctx = build_prompt_and_context(mode, q, chunks, facts)
        out = llm.call(gen_model, prompt, max_tokens=GEN_MAX_TOKENS)
        m = {'answer': out['text'], 'gen_tokens': out['tokens'], 'gen_latency_s': out['latency'],
             'gen_error': out['error'], 'judge': None}
        if out['text']:
            m['judge'] = judge_answer(llm, judge_model, q, ctx, out['text'], item.get('reference'))
        if m['answer'] is None or m['judge'] is None:
            rec['failed'] = True
        m['context_snippet'] = ctx[:600]
        rec['modes'][mode] = m
    return rec


# --------------------------------------------------------------------------- persistence
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    tmp = RESULTS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_FILE)


def run_all(items, pipe, llm, gen_model, judge_model, modes, results, limit=0, retry_failed=False):
    if retry_failed:
        results[:] = [r for r in results if not r.get('failed')]
    done = {r['id'] for r in results}
    todo = [it for it in items if it['id'] not in done]
    if limit:
        todo = todo[:limit]
    for n, item in enumerate(todo, 1):
        print('\n[%d/%d] %s (%s): %s' % (n, len(todo), item['id'], item['category'], item['question'][:80]), flush=True)
        try:
            rec = evaluate_question(item, pipe, llm, gen_model, judge_model, modes)
        except DailyLimit as e:
            print('\nDaily token limit reached. Progress is saved; run the same command again after the limit resets.')
            print('Groq said: %s' % str(e)[:300])
            break
        results.append(rec)
        save_results(results)
        bits = []
        for mode in modes:
            j = rec['modes'][mode]['judge']
            bits.append('%s F=%s R=%s' % (mode, j['faithfulness'] if j else 'x', j['relevance'] if j else 'x'))
        print('  done: %s | graph facts: %d%s' % (' | '.join(bits), rec['num_graph_facts'],
                                                  ' | FAILED parts, rerun with --retry-failed' if rec['failed'] else ''), flush=True)
        print('  tokens used so far: %s' % dict(llm.tokens), flush=True)
    return results


# --------------------------------------------------------------------------- statistics + report
def _vals(v):
    return [x for x in v if x is not None]


def mean(v):
    v = _vals(v)
    return statistics.mean(v) if v else None


def sd(v):
    v = _vals(v)
    return statistics.stdev(v) if len(v) > 1 else None


def median(v):
    v = _vals(v)
    return statistics.median(v) if v else None


def fmt(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def pm(v, nd=2):
    m, s = mean(v), sd(v)
    if m is None:
        return '-'
    return '%s ± %s' % (fmt(m, nd), fmt(s, nd)) if s is not None else fmt(m, nd)


def judged(m):
    return bool(m) and bool(m.get('judge')) and m['judge'].get('faithfulness') is not None


def collect(results, mode, key, subset=None, category=None):
    out = []
    for r in results:
        if subset and r['set'] != subset:
            continue
        if category and r['category'] != category:
            continue
        m = r['modes'].get(mode)
        if judged(m):
            out.append(m['judge'].get(key))
    return out


def halluc_rate(results, mode):
    tot = uns = 0
    for r in results:
        m = r['modes'].get(mode)
        if judged(m) and m['judge'].get('total_claims') is not None and m['judge'].get('unsupported_claims') is not None:
            tot += m['judge']['total_claims']
            uns += m['judge']['unsupported_claims']
    return (uns / tot if tot else None), tot


def paired(results, key, a='hybrid', b='vector', iters=5000, seed=0):
    diffs = []
    for r in results:
        ma, mb = r['modes'].get(a), r['modes'].get(b)
        if judged(ma) and judged(mb) and ma['judge'].get(key) is not None and mb['judge'].get(key) is not None:
            diffs.append(ma['judge'][key] - mb['judge'][key])
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return {'n': n, 'mean': sum(diffs) / n, 'lo': means[int(0.025 * iters)], 'hi': means[int(0.975 * iters) - 1],
            'win': sum(1 for d in diffs if d > 0), 'tie': sum(1 for d in diffs if d == 0),
            'loss': sum(1 for d in diffs if d < 0)}


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def build_report(results):
    L = []
    modes = [m for m in MODES_ALL if any(m in r['modes'] for r in results)]
    hand = [r for r in results if r['set'] == 'hand']
    synth = [r for r in results if r['set'] == 'synthetic']
    gens = sorted({r.get('gen_model') for r in results if r.get('gen_model')})
    judges = sorted({r.get('judge_model') for r in results if r.get('judge_model')})
    L.append('# Automated Evaluation Report')
    L.append('')
    L.append('- Questions evaluated: %d (hand-written: %d, synthetic: %d)' % (len(results), len(hand), len(synth)))
    L.append('- Answer-generation model: %s | Judge model: %s' % (', '.join(gens) or '-', ', '.join(judges) or '-'))
    L.append('- Scores are 1-5 (higher is better); mean ± standard deviation over questions.')
    failed = [r['id'] for r in results if r.get('failed')]
    L.append('- Questions with failed parts (excluded from the averages, not counted as zero): %s' % (', '.join(failed) or 'none'))
    L.append('')

    L.append('## 1. Retrieval quality')
    L.append('')
    if hand:
        keys = [('precision_at_5', 'Precision@5'), ('recall_at_5', 'Recall@5'), ('mrr', 'MRR'),
                ('ndcg_at_5', 'nDCG@5'), ('hit_rate_at_5', 'Hit@5')]
        row = [fmt(mean([(r['retrieval'] or {}).get(k) for r in hand]), 3) for k, _ in keys]
        L.append('**Hand-written questions** (passage relevance judged by the LLM, top-10 judged, n=%d)' % len(hand))
        L.append('')
        L.append(md_table([n for _, n in keys], [row]))
        L.append('')
        cats = sorted({r['category'] for r in hand})
        rows = []
        for c in cats:
            sub = [r for r in hand if r['category'] == c]
            rows.append([c, len(sub)] + [fmt(mean([(r['retrieval'] or {}).get(k) for r in sub]), 3) for k, _ in keys])
        L.append(md_table(['Category', 'n'] + [n for _, n in keys], rows))
        L.append('')
    if synth:
        keys = [('hit_rate_at_5', 'Hit@5'), ('hit_rate_at_10', 'Hit@10'), ('mrr', 'MRR'), ('ndcg_at_5', 'nDCG@5')]
        row = [fmt(mean([(r['retrieval'] or {}).get(k) for r in synth]), 3) for k, _ in keys]
        L.append('**Synthetic questions** (objective: the source chunk must be retrieved, n=%d)' % len(synth))
        L.append('')
        L.append(md_table([n for _, n in keys], [row]))
        L.append('')
    L.append('Note: hybrid and vector-only use the same top-k vector retrieval; the knowledge graph adds extra facts '
             'on top, so retrieval metrics describe the vector stage.')
    L.append('')

    L.append('## 2. Answer quality (ablation: vector-only vs KG-only vs hybrid)')
    L.append('')
    rows = []
    for mode in modes:
        hr, tot = halluc_rate(results, mode)
        n = len(collect(results, mode, 'faithfulness'))
        rows.append([MODE_LABEL[mode], n, pm(collect(results, mode, 'faithfulness')),
                     pm(collect(results, mode, 'relevance')),
                     pm(collect(results, mode, 'correctness')),
                     ('%.1f%% (%d claims)' % (100 * hr, tot)) if hr is not None else '-'])
    L.append(md_table(['System', 'n', 'Faithfulness', 'Relevance', 'Correctness*', 'Hallucination rate**'], rows))
    L.append('')
    L.append('\\* Correctness is measured against a reference answer, so it exists only for synthetic questions.  ')
    L.append('\\** Share of individual claims that the judge found unsupported by the retrieved context (lower is better).')
    L.append('')
    if hand:
        cats = sorted({r['category'] for r in hand})
        rows = []
        for c in cats:
            row = [c]
            for mode in modes:
                row.append('%s / %s' % (fmt(mean(collect(results, mode, 'faithfulness', 'hand', c))),
                                        fmt(mean(collect(results, mode, 'relevance', 'hand', c)))))
            rows.append(row)
        L.append('**By question category, hand-written set (Faithfulness / Relevance)**')
        L.append('')
        L.append(md_table(['Category'] + [MODE_LABEL[m] for m in modes], rows))
        L.append('')

    if 'hybrid' in modes and 'vector' in modes:
        L.append('## 3. Hybrid vs vector-only (paired, per question)')
        L.append('')
        rows = []
        for key, name in [('faithfulness', 'Faithfulness'), ('relevance', 'Relevance'), ('correctness', 'Correctness')]:
            p = paired(results, key)
            if p:
                rows.append([name, p['n'], '%+.2f' % p['mean'], '[%+.2f, %+.2f]' % (p['lo'], p['hi']),
                             '%d / %d / %d' % (p['win'], p['tie'], p['loss'])])
        if rows:
            L.append(md_table(['Metric', 'n', 'Mean difference (hybrid − vector)', '95% bootstrap CI', 'Wins / ties / losses'], rows))
            L.append('')
            L.append('A confidence interval that excludes 0 means the difference is unlikely to be chance.')
            L.append('')

    L.append('## 4. Efficiency and knowledge-graph coverage')
    L.append('')
    cov = [1.0 if r.get('num_graph_facts', 0) > 0 else 0.0 for r in results]
    L.append('- Questions where the knowledge graph supplied at least one fact: %.0f%% (mean %.1f facts per question)'
             % (100 * mean(cov), mean([r.get('num_graph_facts', 0) for r in results])))
    L.append('- Median vector retrieval time: %s ms | median KG traversal time: %s ms'
             % (fmt(median([r['timing']['retrieval_ms'] for r in results]), 0),
                fmt(median([r['timing'].get('kg_traversal_ms') for r in results]), 1)))
    L.append('')
    rows = []
    for mode in modes:
        lat = [r['modes'][mode]['gen_latency_s'] for r in results if mode in r['modes'] and r['modes'][mode]['answer']]
        tok = [r['modes'][mode]['gen_tokens'] for r in results if mode in r['modes'] and r['modes'][mode]['answer']]
        rows.append([MODE_LABEL[mode], fmt(median(lat), 1), fmt(mean(tok), 0)])
    L.append(md_table(['System', 'Median generation latency (s)', 'Mean tokens per answer (prompt+completion)'], rows))
    L.append('')

    L.append('## 5. Validity notes and limitations')
    L.append('')
    L.append('- Judging is done by an LLM. See judge_audit.csv: a human spot-check of a sample, summarised by `--agreement`.')
    L.append('- Generator and judge are from the same model family, which can favour similar phrasing. Different model sizes are used for the two roles.')
    L.append('- The generator and the judge see identical context text (no silent truncation of the knowledge-graph facts).')
    L.append('- Synthetic questions are written from a single chunk, so they favour lexical/vector retrieval; the hand-written set contains the relational and multi-document questions where the graph should help.')
    L.append('- The synthetic gold-chunk metric is strict: a near-duplicate chunk of the same content counts as a miss.')
    L.append('- Small sample sizes: use the confidence intervals, not just the means.')
    return '\n'.join(L)


def write_csv(results):
    modes = [m for m in MODES_ALL if any(m in r['modes'] for r in results)]
    cols = ['id', 'set', 'category', 'question', 'num_graph_facts', 'p_at_5', 'r_at_5', 'mrr', 'ndcg_at_5', 'hit_at_5']
    for m in modes:
        cols += ['%s_faith' % m, '%s_rel' % m, '%s_corr' % m, '%s_claims' % m, '%s_unsupported' % m, '%s_tokens' % m]
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            ret = r.get('retrieval') or {}
            row = [r['id'], r['set'], r['category'], r['question'], r.get('num_graph_facts'),
                   ret.get('precision_at_5'), ret.get('recall_at_5'), ret.get('mrr'), ret.get('ndcg_at_5'), ret.get('hit_rate_at_5')]
            for m in modes:
                mm = r['modes'].get(m) or {}
                j = mm.get('judge') or {}
                row += [j.get('faithfulness'), j.get('relevance'), j.get('correctness'),
                        j.get('total_claims'), j.get('unsupported_claims'), mm.get('gen_tokens')]
            w.writerow(row)


def write_audit(results, n=12, seed=1):
    if os.path.exists(AUDIT_FILE):
        return False
    pairs = []
    for r in results:
        for mode in ('vector', 'hybrid'):
            m = r['modes'].get(mode)
            if judged(m):
                pairs.append((r, mode))
    if not pairs:
        return False
    random.Random(seed).shuffle(pairs)
    with open(AUDIT_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'mode', 'question', 'context_snippet', 'answer', 'judge_faithfulness', 'judge_relevance',
                    'human_faithfulness', 'human_relevance'])
        for r, mode in pairs[:n]:
            m = r['modes'][mode]
            w.writerow([r['id'], mode, r['question'], m['context_snippet'], (m['answer'] or '')[:900],
                        m['judge']['faithfulness'], m['judge']['relevance'], '', ''])
    return True


def write_report(results):
    if not results:
        print('No results yet.')
        return
    text = build_report(results)
    with open(REPORT_FILE, 'w') as f:
        f.write(text + '\n')
    write_csv(results)
    created = write_audit(results)
    print(text)
    print('\nSaved: %s, %s, %s' % (REPORT_FILE, CSV_FILE, AUDIT_FILE if os.path.exists(AUDIT_FILE) else '(no audit file)'))
    if created:
        print('Next: open %s, fill human_faithfulness and human_relevance (1-5) for each row, then run --agreement.' % AUDIT_FILE)


def agreement():
    if not os.path.exists(AUDIT_FILE):
        print('No %s yet. Run --report first.' % AUDIT_FILE)
        return
    rows = list(csv.DictReader(open(AUDIT_FILE)))
    for dim in ('faithfulness', 'relevance'):
        pairs = []
        for r in rows:
            h, j = (r.get('human_' + dim) or '').strip(), (r.get('judge_' + dim) or '').strip()
            if h and j:
                pairs.append((int(float(h)), int(float(j))))
        if not pairs:
            print('%s: no human scores filled in yet' % dim)
            continue
        n = len(pairs)
        exact = sum(1 for h, j in pairs if h == j) / n
        within1 = sum(1 for h, j in pairs if abs(h - j) <= 1) / n
        mae = sum(abs(h - j) for h, j in pairs) / n
        print('%-13s n=%d  exact agreement %.0f%%  within ±1 point %.0f%%  mean abs. difference %.2f'
              % (dim, n, 100 * exact, 100 * within1, mae))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Automated evaluation of the Cisco KG-RAG assistant')
    ap.add_argument('--only', choices=['hand', 'synthetic', 'all'], default='all')
    ap.add_argument('--n-synth', type=int, default=12)
    ap.add_argument('--modes', default=','.join(MODES_ALL))
    ap.add_argument('--limit', type=int, default=0, help='process only the first N unfinished questions')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--agreement', action='store_true')
    args = ap.parse_args()

    if args.agreement:
        agreement()
        return
    results = load_results()
    if args.report:
        write_report(results)
        return

    modes = [m for m in args.modes.split(',') if m in MODES_ALL]
    if not os.environ.get('GROQ_API_KEY'):
        print('GROQ_API_KEY is not set in this terminal. Run: source ~/.zshrc')
        raise SystemExit(1)
    from groq import Groq
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    gen_model = pick_model(client, GEN_MODEL_PREF, os.environ.get('GEN_MODEL'), 'generation')
    judge_model = pick_model(client, JUDGE_MODEL_PREF, os.environ.get('JUDGE_MODEL'), 'judging')
    print('Generation model: %s | Judge model: %s | TPM budget per model: %d' % (gen_model, judge_model, TPM_LIMIT), flush=True)
    llm = LLM(client)
    pipe = Pipeline()

    items = []
    if args.only in ('hand', 'all'):
        items += [{'id': i, 'set': 'hand', 'category': c, 'question': q} for i, c, q in HAND_QUESTIONS]
    try:
        if args.only in ('synthetic', 'all'):
            items += build_synthetic(pipe, llm, judge_model, args.n_synth)
        run_all(items, pipe, llm, gen_model, judge_model, modes, results, args.limit, args.retry_failed)
    except FatalError as e:
        print('\nStopped: %s' % str(e)[:300])
        print('Fix the key/model problem above, then run the same command again (progress is saved).')
    except DailyLimit as e:
        print('\nDaily token limit reached while preparing questions. Run the same command again after it resets.')
        print(str(e)[:300])
    write_report(results)


if __name__ == '__main__':
    main()
