# ==========================================
# bullshit_agent_langgraph.py  (corrected)
# ==========================================

from typing import TypedDict, Annotated, List, Optional
import faiss
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import json
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from openai import OpenAI
from inference_auth_token import get_access_token

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI

# -----------------------------
# 1. MODEL INITIALIZATION
# -----------------------------

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

nli_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
nli_model.to(device)

# LLM / agent model (via LangChain + LangGraph)
access_token = get_access_token()
chat = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_key=access_token,
    openai_api_base="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
)

# -----------------------------
# 2. USER MEMORY / FAISS
# -----------------------------

user_data = {}
user_indices = {}

def get_or_create_index(user_id: str):
    if user_id not in user_indices:
        user_indices[user_id] = faiss.IndexFlatIP(768)
    return user_indices[user_id]

def add_user_texts(user_id: str, texts: List[str]):
    if user_id not in user_data:
        user_data[user_id] = {"texts": []}
    idx = get_or_create_index(user_id)
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    idx.add(np.array(embeddings, dtype="float32"))
    user_data[user_id]["texts"].extend(texts)

def retrieve_similar(user_id: str, draft: str, top_k: int = 5) -> List[str]:
    if user_id not in user_data:
        return []
    idx = get_or_create_index(user_id)
    draft_emb = embedder.encode([draft], normalize_embeddings=True)
    D, I = idx.search(np.array(draft_emb, dtype="float32"), top_k)
    texts = user_data[user_id]["texts"]
    return [texts[i] for i in I[0] if i < len(texts)]

# -----------------------------
# 3. EXTERNAL CONTEXT TOOLS
# -----------------------------

def fetch_wikipedia(query: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("extract", "")
    except Exception:
        pass
    return ""

def fetch_bbc(query: str, max_results: int = 2) -> List[str]:
    url = f"https://www.bbc.co.uk/search?q={query}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for a in soup.select("article h1 a")[:max_results]:
            href = a.get("href", "")
            headline = a.get_text(strip=True)
            try:
                art = requests.get(href, timeout=5)
                art_soup = BeautifulSoup(art.text, "html.parser")
                paras = " ".join(p.get_text() for p in art_soup.find_all("p")[:3])
                results.append(f"{headline}: {paras}")
            except Exception:
                continue
        return results
    except Exception:
        return []

def get_external_context(draft: str) -> List[str]:
    terms = draft.split()[:5]
    query = " ".join(terms)
    wiki = fetch_wikipedia(query)
    bbc = fetch_bbc(query)
    return ([wiki] if wiki else []) + bbc

# -----------------------------
# 4. NLI SCORING
# -----------------------------

def nli_score(premise: str, hypothesis: str):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    return {
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2]),
    }

# -----------------------------
# Helpers
# -----------------------------

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract and parse a JSON object from text.
    Handles:
      - raw JSON
      - ```json ... ```
      - ``` ... ```
      - or first {...} object found in text
    Returns parsed dict or None.
    """
    if text is None:
        return None

    # 1. Try direct JSON first (trim whitespace)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # 2. Extract from triple backticks (with optional "json" tag)
    pattern = r"```(?:json)?\s*({.*?})\s*```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        block = match.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass

    # 3. Fallback: extract first {...} JSON-looking object anywhere
    brace_pattern = r"(\{(?:[^{}]|\{.*?\})*\})"
    m = re.search(brace_pattern, text, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

# -----------------------------
# 5. LangGraph State Definition
# -----------------------------

class BullshitState(TypedDict):
    user_id: str
    draft: str
    user_contexts: Annotated[List[str], "texts from user history"]
    external_contexts: Annotated[List[str], "wiki / news sources"]
    combined_contexts: List[str]
    nli_contradictions: List[str]
    nli_scores: List[dict]
    llm_response: object  # will store parsed dict or raw string

# -----------------------------
# 6. Nodes for Graph
# -----------------------------

def iterative_context_node(state: BullshitState) -> BullshitState:
    """
    Iteratively determine what context is needed, fetch it,
    evaluate whether more is required, and repeat until the LLM
    declares context is sufficient.
    """
    accumulated_context: List[str] = []
    max_rounds = 3

    for round_idx in range(max_rounds):
        print(f"round: {round_idx}")

        # 1) Ask LLM what it needs
        prompt = f"""
You are assisting with fact-checking and grounding a draft tweet.

Draft:
\"\"\"{state['draft']}\"\"\"


Current external context (may be empty):
{accumulated_context}

Your task:
- Determine what additional real-world information is necessary.
- Ensure at least one (1) outside source is present
- Return a JSON object with either:
    {{ "queries": ["query1", "query2", ...] }}
  or
    {{ "done": true }}
Respond ONLY with JSON. Example:
{{ "queries": ["robotics labor market", "AI art displacement"] }}
"""
        resp = chat.invoke([{"role": "user", "content": prompt}])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = extract_json_from_text(raw)

        print(f"parsed queries response: {parsed}")

        if not parsed:
            # couldn't parse JSON from the LLM; stop the loop to avoid spinning
            print("breaking (could not parse LLM planning JSON)")
            break

        if parsed.get("done"):
            break

        queries = parsed.get("queries", [])
        if not queries:
            print("breaking (no queries returned)")
            break

        # 2) Run the lookups for each query
        new_context_chunks: List[str] = []
        for q in queries:
            print(f"running query: {q}")
            wiki = fetch_wikipedia(q)
            bbc_items = fetch_bbc(q)
            print(f"wiki: {wiki}")
            if wiki:
                print(f"wiki: {q} -> (len {len(wiki)})")
                new_context_chunks.append(f"Wikipedia({q}): {wiki}")
            for item in bbc_items:
                print(f"bbc: {q} -> (len {len(item)})")
                new_context_chunks.append(f"BBC({q}): {item}")

        # 3) Add to accumulated context
        accumulated_context.extend(new_context_chunks)

        print(f"accumulated_content: {accumulated_context}")

        # 4) Ask LLM if this context is sufficient
        suff_prompt = f"""
Draft:
\"\"\"{state['draft']}\"\"\"


Accumulated context:
{accumulated_context}

Question:
Do you now have enough context to assess the draft?
Respond ONLY with JSON:
- {{ "sufficient": true }} 
or
- {{ "sufficient": false }}
"""
        suff_resp = chat.invoke([{"role": "user", "content": suff_prompt}])
        suff_raw = suff_resp.content if hasattr(suff_resp, "content") else str(suff_resp)
        suff_parsed = extract_json_from_text(suff_raw)

        print(f"suff_parsed: {suff_parsed}")

        if suff_parsed and suff_parsed.get("sufficient"):
            break
        # otherwise continue to next round

    # Store the final set of contexts in the state
    state["external_contexts"] = accumulated_context
    # ensure user_contexts exists
    if "user_contexts" not in state or state["user_contexts"] is None:
        state["user_contexts"] = []
    state["combined_contexts"] = state["user_contexts"] + accumulated_context
    return state

def nli_node(state: BullshitState) -> BullshitState:
    scores = []
    contradictions = []
    for ctx in state["combined_contexts"]:
        sc = nli_score(ctx, state["draft"])
        scores.append(sc)
        if sc["contradiction"] > 0.6:
            contradictions.append(ctx)
    state["nli_scores"] = scores
    state["nli_contradictions"] = contradictions
    return state

def llm_assess_node(state: BullshitState) -> BullshitState:
    # Build a prompt
    ctx_for_prompt = "\n".join(f"- {c}" for c in state["combined_contexts"][:10])
    print(f"Context: {state['combined_contexts']}")
    prompt = f"""
You are a thoughtful assistant analyzing a tweet draft.

Draft Tweet:
\"\"\"{state['draft']}\"\"\"


Context:
{ctx_for_prompt}

Please analyze:
- Whether the draft contradicts the user's prior statements
- Whether the draft is supported or contradicted by factual sources
- Whether it's vague or overconfident (i.e., "bullshit")
- Suggest a more grounded version if needed

Respond with a JSON object:
{{ 
  "summary": "...", 
  "risk": "low/medium/high",
  "reasoning": "...",
  "suggested_revision": "..." 
}}
"""
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    if parsed is None:
        # fallback: store raw answer so downstream code can inspect it
        print("Warning: could not parse assessment JSON from LLM. Storing raw text.")
        state["llm_response"] = raw
    else:
        state["llm_response"] = parsed
    return state

# -----------------------------
# 7. Build & Run the Graph
# -----------------------------

graph = StateGraph(BullshitState)
graph.add_node("retrieve", iterative_context_node)
graph.add_node("nli", nli_node)
graph.add_node("assess", llm_assess_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "nli")
graph.add_edge("nli", "assess")
graph.add_edge("assess", END)

graph = graph.compile()

# -----------------------------
# 8. High-level API
# -----------------------------

def run_bullshit_graph(user_id: str, draft: str):
    # ensure user data present
    if user_id not in user_data:
        user_data[user_id] = {"texts": []}
    state0: BullshitState = {
        "user_id": user_id,
        "draft": draft,
        "user_contexts": [],
        "external_contexts": [],
        "combined_contexts": [],
        "nli_contradictions": [],
        "nli_scores": [],
        "llm_response": "",
    }
    result = graph.invoke(state0)
    return result

# -----------------------------
# 9. Example / Demo
# -----------------------------

if __name__ == "__main__":
    uid = "user1"
    user_data[uid] = {"texts": [
        "AI helps artists make new kinds of art.",
        "I believe automation should help humans, not replace them."
    ]}
    add_user_texts(uid, user_data[uid]["texts"])

    draft = "AI will replace all artists soon."
    out = run_bullshit_graph(uid, draft)

    print("LLM response:", out["llm_response"])
    print("Contradictory contexts:", out["nli_contradictions"])
