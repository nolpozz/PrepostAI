# ==========================================
# bullshit_agent_rag.py  (Local MVP version)
# ==========================================

import faiss
import numpy as np
import torch
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup

# -----------------------------
# 1. MODEL INITIALIZATION
# -----------------------------

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

nli_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
nli_model.to(device)

label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

# -----------------------------
# 2. SIMPLE IN-MEMORY DATABASE
# -----------------------------

user_data = {
    # user_id : {
    #   "tweets": [list of strings],
    #   "articles": [list of urls]
    # }
}

# store FAISS indices per user
user_indices = {}

def get_or_create_index(user_id):
    if user_id not in user_indices:
        user_indices[user_id] = faiss.IndexFlatIP(768)
    return user_indices[user_id]

def add_user_texts(user_id, texts):
    """Embed and add texts (tweets/articles) to user's FAISS index"""
    index = get_or_create_index(user_id)
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    index.add(np.array(embeddings, dtype="float32"))
    user_data[user_id]["tweets"].extend(texts)

# -----------------------------
# 3. RETRIEVAL PIPELINE
# -----------------------------

def retrieve_similar(user_id, draft, top_k=5):
    """Retrieve semantically similar prior tweets/articles"""
    index = get_or_create_index(user_id)
    draft_emb = embedder.encode([draft], normalize_embeddings=True)
    D, I = index.search(np.array(draft_emb, dtype="float32"), top_k)
    # Get all user texts for reference
    texts = user_data[user_id]["tweets"]
    results = [texts[i] for i in I[0] if i < len(texts)]
    return results

# -----------------------------
# 4. WIKIPEDIA + BBC RETRIEVAL
# -----------------------------

def fetch_wikipedia(query, sentences=3):
    """Simple Wikipedia search API call"""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return data.get("extract", "")
    return ""

def fetch_bbc(query, max_results=2):
    """Lightweight BBC News scrape (headline + snippet)"""
    search_url = f"https://www.bbc.co.uk/search?q={query}"
    r = requests.get(search_url)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    articles = []
    for a in soup.select("article h1 a")[:max_results]:
        href = a["href"]
        headline = a.get_text(strip=True)
        try:
            art = requests.get(href, timeout=5)
            art_soup = BeautifulSoup(art.text, "html.parser")
            paras = " ".join(p.get_text() for p in art_soup.find_all("p")[:3])
            articles.append(headline + ": " + paras)
        except Exception:
            continue
    return articles

def get_external_context(draft):
    """Aggregate Wikipedia + BBC snippets related to the draft"""
    key_terms = draft.split()[:5]  # crude heuristic
    query = " ".join(key_terms)
    wiki_text = fetch_wikipedia(query)
    bbc_articles = fetch_bbc(query)
    return [wiki_text] + bbc_articles

# -----------------------------
# 5. NLI SCORING
# -----------------------------

def nli_score(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    return {
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2])
    }

# -----------------------------
# 6. BULLSHIT SCORING
# -----------------------------

def bullshit_score(user_id, draft, top_k=5):
    """Evaluate self-consistency and factual grounding"""
    user_contexts = retrieve_similar(user_id, draft, top_k)
    external_contexts = get_external_context(draft)

    total_contra, total_entail = 0, 0
    all_contexts = user_contexts + external_contexts

    for ctx in all_contexts:
        nli = nli_score(premise=ctx, hypothesis=draft)
        total_contra += nli["contradiction"]
        total_entail += nli["entailment"]

    contra_avg = total_contra / len(all_contexts)
    entail_avg = total_entail / len(all_contexts)
    score = (contra_avg - entail_avg + 1) / 2  # 0–1 normalized

    contradictions = [c for c in user_contexts if nli_score(c, draft)["contradiction"] > 0.6]

    feedback = (
        f"⚠️ Possible inconsistency ({score:.2f}). Check against prior tweets or factual sources."
        if score > 0.6 else
        f"✅ Appears consistent ({score:.2f})."
    )

    return {
        "score": score,
        "feedback": feedback,
        "contradictions": contradictions,
        "external_used": external_contexts
    }

# -----------------------------
# 7. DEMO EXECUTION
# -----------------------------

if __name__ == "__main__":
    user_id = "user123"
    user_data[user_id] = {
        "tweets": [
            "AI will empower artists to explore new creative possibilities.",
            "Technology should augment human creativity, not replace it."
        ],
        "articles": []
    }

    # Add tweets to FAISS index
    add_user_texts(user_id, user_data[user_id]["tweets"])

    # Example: user drafts a new tweet
    draft = "AI is replacing all artists and creativity will soon be obsolete."
    result = bullshit_score(user_id, draft)

    print("\n--- Bullshit Analysis ---")
    print("Draft:", draft)
    print(result["feedback"])
    print("\nContradictory self-tweets:")
    for c in result["contradictions"]:
        print("-", c)
    print("\nExternal context snippets:")
    for e in result["external_used"]:
        print("-", e[:200], "...\n")
