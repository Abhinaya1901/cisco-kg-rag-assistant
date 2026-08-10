import os
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Cisco Catalyst 9300 Assistant")

print("Loading pipeline...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
client_db = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "..", "chroma_db"))
collection = client_db.get_collection("cisco_docs")
groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
G = nx.read_gml(os.path.join(BASE_DIR, "..", "knowledge_graph.gml"))
print(f"Loaded. Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

ENTITY_PROMPT = """Extract key technical entities (features, protocols, commands, interfaces) from this question. Return ONLY a JSON list of lowercase strings.

Question: {question}

JSON list:"""

def extract_question_entities(question):
    try:
        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': ENTITY_PROMPT.format(question=question)}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        return json.loads(content)
    except Exception:
        return []

def find_matching_nodes(entity, graph, max_matches=2):
    entity = entity.lower().strip()
    matches = []
    for node in graph.nodes():
        if entity in node or node in entity:
            matches.append(node)
        if len(matches) >= max_matches:
            break
    return matches

def get_graph_facts(question_entities, graph, max_facts=8):
    facts = []
    for entity in question_entities:
        for node in find_matching_nodes(entity, graph):
            for neighbor in list(graph.neighbors(node))[:3]:
                edge_data = graph.get_edge_data(node, neighbor)
                facts.append(f"{node} --{edge_data.get('relation','related_to')}--> {neighbor}")
            if len(facts) >= max_facts:
                return facts
    return facts

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question

    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    retrieved_chunks = results["documents"][0]

    question_entities = extract_question_entities(question)
    graph_facts = get_graph_facts(question_entities, G)

    chunk_context = "\n\n---\n\n".join(retrieved_chunks)
    graph_context = "\n".join(graph_facts) if graph_facts else "No direct graph relationships found."

    prompt = f"""You are a network engineering assistant. Answer the question using the context below.

DOCUMENT EXCERPTS:
{chunk_context}

KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context}

Question: {question}

Provide a complete, accurate answer using both sources above:"""

    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}]
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "graph_facts": graph_facts,
        "num_chunks": len(retrieved_chunks)
    }

# Serve the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
