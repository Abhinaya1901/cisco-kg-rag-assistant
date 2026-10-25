# Catalyst Assistant: Hybrid Knowledge-Graph and Vector RAG for Cisco Catalyst 9300 Documentation

A question-answering assistant for technical documentation. It combines **semantic vector search** (ChromaDB with `all-MiniLM-L6-v2`) with a **knowledge graph** (NetworkX) of entities and relations extracted from the same documents, and asks a large language model to answer **only** from the retrieved material.

The repository contains the working web application and the evaluation scripts that were used to test it, including a controlled comparison of *graph versus prompt* and a second study on 3GPP Release 18 specifications with the TeleQnA benchmark.

> Project: SWE3004 Software Design and Development Project, M.Tech Software Engineering, VIT Vellore.
> Author: Kolla Srinivasan Abhinaya (22MIS0223). Guide: Dr. C. Vanmathi.

---

## Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Quick start](#quick-start)
5. [API](#api)
6. [Data](#data)
7. [Evaluation](#evaluation)
8. [Results at a glance](#results-at-a-glance)
9. [Limitations](#limitations)
10. [Configuration and troubleshooting](#configuration-and-troubleshooting)

---

## What it does

Ask a question in plain language, for example *"What do I need to configure before setting up a trunk port?"* For each question the system:

1. embeds the question and fetches the **5 most similar passages** from the vector index;
2. asks the language model for the **technical entities** in the question and looks them up in the knowledge graph, collecting up to **8 related facts** (`node --relation--> neighbour`);
3. places passages and graph facts in **one prompt** that tells the model to use only that material and to say so when it does not contain the answer;
4. returns the answer together with the graph facts and the number of passages used. The chat page shows these counts under every answer.

## Architecture

```
 Cisco Catalyst 9300 pages (240)
        |  clean, chunk (400 words, 50 overlap)
        v
  2,186 passages --embed (MiniLM, 384-d)--> ChromaDB  (collection: cisco_docs)
        |  LLM triple extraction
        v
  2,619 triples --merge--> NetworkX graph (2,589 nodes, 2,459 edges, knowledge_graph.gml)

 Question --> vector search (top 5) ----------------+
          --> entity extraction --> graph lookup ---+--> one prompt --> LLM (Groq) --> answer
```

| Part | Technology |
|---|---|
| Web service | FastAPI and Uvicorn |
| Vector store | ChromaDB (persistent), `all-MiniLM-L6-v2` embeddings |
| Knowledge graph | NetworkX, stored as GML |
| Language model | `openai/gpt-oss-120b` through the Groq API (generator and entity extractor) |
| Frontend | Plain HTML, CSS and JavaScript served by the backend |

## Repository layout

```
backend/main.py          FastAPI application (retrieval, graph lookup, prompt, endpoints)
frontend/                Chat interface (index.html, script.js, style.css)
chroma_db/               Persistent vector index of the 2,186 passages
knowledge_graph.gml      Knowledge graph (2,589 nodes, 2,459 edges)
triples.json             The 2,619 extracted triples with their source pages
raw_docs.json            Cleaned corpus: 240 pages
requirements.txt         Python dependencies
evaluation/              Automated evaluation scripts (see Evaluation)
```

## Quick start

Requirements: Python 3.10 or newer (developed on 3.12) and a free [Groq API key](https://console.groq.com/).

```bash
git clone https://github.com/Abhinaya1901/cisco-kg-rag-assistant.git
cd cisco-kg-rag-assistant

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=<your key>       # Windows PowerShell: $env:GROQ_API_KEY="<your key>"

cd backend
uvicorn main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. The first start downloads the MiniLM model (about 90 MB). No GPU is needed.

The key is read from the environment and is never stored in the code or in any result file.

## API

### `GET /api/health`

```json
{ "status": "ok", "graph_nodes": 2589, "graph_edges": 2459 }
```

### `POST /api/chat`

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the default VTP mode?"}'
```

```json
{
  "answer": "...",
  "graph_facts": ["vlan configuration --requires--> vlan trunking protocol (vtp) decision", "..."],
  "num_chunks": 5
}
```

| Status | Meaning |
|---|---|
| 200 | Answer generated |
| 400 | Empty question |
| 502 | The language-model call failed (rate limit, invalid key, model unavailable) |

If entity extraction fails, the answer is produced from the passages alone.

## Data

| Corpus | Source | Size |
|---|---|---|
| Cisco Catalyst 9300 | Public Cisco support documentation (configuration guides, command references, troubleshooting, release notes, datasheet) | 240 pages, 729,580 words, 2,186 passages |
| 3GPP Release 18 | Latest Release 18 versions of 25 core specifications from the public 3GPP file server | 25,377 passages |
| TeleQnA | Public telecommunications question benchmark ([paper](https://arxiv.org/abs/2310.15051)) | 780 Release 18 questions, 395 answerable from the downloaded specifications |

The 3GPP passages and the TeleQnA questions are **not** included in this repository. Download them from their public sources. The 3GPP study uses a graph built without a language model, from cross-references between clauses (10,056 edges) and continuation links between neighbouring passages (16,185 edges).

## Evaluation

Everything is scored automatically, with no manual rating. The scripts are in `evaluation/` and are meant to be run **from the project root** (the folder that contains `chroma_db/` and `knowledge_graph.gml`):

```bash
export GROQ_API_KEY=<your key>

# 1. Cisco: retrieval metrics, judged answer quality, hallucination rate, paired bootstrap
python3 -u evaluation/evaluation_v2.py --only hand --limit 1      # smoke test
python3 -u evaluation/evaluation_v2.py                            # 18 hand-written + 12 generated questions
python3 -u evaluation/evaluation_v2.py --report                   # rebuild the report from saved results

# 2. Controlled prompt ablation, similarity metrics, second judge
python3 -u evaluation/evaluation_v3.py

# 3. Sentence-level audit (no LLM calls)
python3 -u evaluation/audit_hallucination.py

# 4. 3GPP / TeleQnA (needs the corpus files described in each script's header)
python3 -u evaluation/evaluation_tele.py --n 200
python3 -u evaluation/tele_retrieval_eval.py --n 200
python3 -u evaluation/retrieval_improve.py --n 200
```

| Script | What it does |
|---|---|
| `evaluation_v2.py` | Pipeline, rate-limit aware LLM wrapper, question generation, Precision@5, Recall@5, MRR, nDCG@5, Hit@k, LLM judge (faithfulness, relevance, correctness, claim-level hallucination), paired bootstrap |
| `evaluation_v3.py` | Strict-prompt hybrid system, reference-word recall and embedding cosine, second judge and agreement |
| `audit_hallucination.py` | Sentence-level check of every answer against its context, with no language model |
| `evaluation_tele.py` | The same evaluation on TeleQnA with a structural graph, open-ended and multiple-choice |
| `tele_retrieval_eval.py` | Retrieval metrics on 3GPP and the contribution of the structural graph |
| `retrieval_improve.py` | BM25, rank fusion, BGE-small and cross-encoder re-ranking |

The scripts save progress after every question and resume after interruptions. Sampling uses seed 42 and the bootstrap uses 5,000 resamples with seed 0, so reports can be reproduced. Free-tier Groq limits apply; set `TPM_LIMIT` if your limit differs.

## Results at a glance

30 Cisco questions (18 hand-written, 12 generated), four systems, same generator:

| System | Faithfulness | Relevance | Correctness (n = 12) | Unsupported claims |
|---|---|---|---|---|
| Vector only | 4.40 | 4.67 | 4.17 | 9.3% (of 161) |
| Knowledge graph only | 2.53 | 2.33 | 1.00 | 54.1% (of 37) |
| Hybrid, original prompt | 4.17 | 4.87 | 4.50 | 21.4% (of 187) |
| Hybrid, strict prompt | 4.67 | 4.70 | 4.33 | 10.8% (of 158) |

* The instruction to use **only** the supplied context raised faithfulness by **+0.50** (95% interval +0.13 to +0.93) and roughly halved the unsupported-claim rate. It is the only change on Cisco whose interval excludes zero.
* With matched prompts, hybrid and vector-only did **not** differ significantly (faithfulness +0.27, interval −0.27 to +0.80). The graph adds about 1.4 ms and about 7% more tokens per answer.
* Retrieval put a relevant passage in the top five for 83% of the Cisco questions.
* On 3GPP Release 18 with TeleQnA, retrieval raised multiple-choice accuracy from 63.5% to 74.0% (+10.5 points, interval +3.0 to +18.5). Combining BM25, MiniLM and a cross-encoder raised Hit@5 from 0.600 to 0.750. The structural graph rescued only 5 to 7 of 200 questions.

The honest summary is that **the prompt and the retriever mattered more than the graph** in these experiments.

## Limitations

* Small samples: 30 Cisco questions, 200 per TeleQnA test. A difference that is not significant is undecided, not proven equal.
* The generator and the main judge are both `gpt-oss` models. A second judge from another family reduces self-preference bias but no human scoring was done.
* Triples are extracted by a language model and can be noisy. Graph lookup is a substring match with one-hop neighbours.
* Generated questions are written from a single passage, which favours vector search.
* One product family and a subset of the 3GPP specifications.
* No automated unit tests for the web service.
* Answers are grounded in the documentation but must still be verified before they are applied to production equipment.

## Configuration and troubleshooting

| Setting | Where | Default |
|---|---|---|
| `GROQ_API_KEY` | environment variable | required |
| Generation model | `MODEL` in `backend/main.py` | `openai/gpt-oss-120b` |
| Passages per question | `n_results` in `backend/main.py` | 5 |
| Graph facts per question | `max_facts` in `get_graph_facts` | 8 |
| `TPM_LIMIT`, `CHUNK_CHARS` | environment variables for the evaluation | 6000, 1000 |

* **HTTP 502 from `/api/chat`**: the model call failed. Check the key, the Groq rate limit and that the model name is still served (the first version used `llama-3.3-70b-versatile`, which stopped being served and returned 404).
* **Slow first start**: the embedding model is downloading.
* **Low memory hosts**: `main.py` sets `OMP_NUM_THREADS` and `TORCH_NUM_THREADS` to 1 to reduce memory use.

## Acknowledgements

Cisco and 3GPP documentation are the property of their owners and are used here for research and education. TeleQnA is by Maatouk et al. (2023).
