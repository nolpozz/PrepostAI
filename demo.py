# ==========================================
# demo_bullshit_detector.py
# Interactive demo of the three architectural approaches
# ==========================================

from typing import TypedDict, Annotated, List, Optional, Dict, Any
import faiss
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import json
import re
import time

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from openai import OpenAI
from inference_auth_token import get_access_token

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

print("Loading models... (this may take a moment)")

# -----------------------------
# MODEL INITIALIZATION
# -----------------------------

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

nli_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
nli_model.to(device)

access_token = get_access_token()
chat = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_key=access_token,
    openai_api_base="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
)

# -----------------------------
# FAKE USER HISTORY (Hardcoded)
# -----------------------------

DEMO_USER_HISTORY = [
    "Climate change is one of the biggest challenges facing humanity",
    "I believe in evidence-based policy making",
    "Renewable energy is becoming more cost-effective every year",
    "We need to invest in green technology",
    "Science and data should guide our climate policies",
    "Electric vehicles are the future of transportation",
    "Carbon emissions need to be reduced dramatically",
    "I support international cooperation on climate action",
    "Technology can help us solve environmental problems",
    "We must listen to climate scientists",
]

DEMO_USER_ID = "demo_user"

# -----------------------------
# USER MEMORY / FAISS
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
    
    if not texts:
        return
    
    idx = get_or_create_index(user_id)
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    idx.add(np.array(embeddings, dtype="float32"))
    user_data[user_id]["texts"].extend(texts)

def retrieve_similar(user_id: str, draft: str, top_k: int = 5) -> List[str]:
    if user_id not in user_data:
        return []
    idx = get_or_create_index(user_id)
    if idx.ntotal == 0:
        return []
    draft_emb = embedder.encode([draft], normalize_embeddings=True)
    D, I = idx.search(np.array(draft_emb, dtype="float32"), min(top_k, idx.ntotal))
    texts = user_data[user_id]["texts"]
    return [texts[i] for i in I[0] if i < len(texts)]

# Initialize demo user history
add_user_texts(DEMO_USER_ID, DEMO_USER_HISTORY)

# -----------------------------
# SEARCH TOOLS
# -----------------------------

def fetch_wikipedia(query: str) -> str:
    """Fetch Wikipedia summary for a query."""
    print(f"   🔍 Searching Wikipedia for: '{query}'")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            print(f"   ✅ Found Wikipedia article ({len(extract)} chars)")
            return extract
        else:
            # Try search API
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            search_r = requests.get(search_url, headers=headers, timeout=10)
            if search_r.status_code == 200:
                search_data = search_r.json()
                results = search_data.get("query", {}).get("search", [])
                if results:
                    snippet = results[0].get("snippet", "")
                    snippet = re.sub('<.*?>', '', snippet)
                    print(f"   ✅ Found Wikipedia snippet ({len(snippet)} chars)")
                    return snippet
    except Exception as e:
        print(f"   ❌ Wikipedia error: {e}")
    return ""

def fetch_bbc(query: str, max_results: int = 2) -> List[str]:
    """Fetch BBC news articles for a query."""
    print(f"   🔍 Searching BBC News for: '{query}'")
    url = f"https://www.bbc.co.uk/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code != 200:
            print(f"   ❌ BBC returned status {r.status_code}")
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        
        articles = (soup.select("article a[href*='/news/']") or 
                   soup.select("a[href*='/news/']"))
        
        seen_urls = set()
        for a in articles[:max_results * 3]:
            href = a.get("href", "")
            if not href or href in seen_urls:
                continue
            if "/news/" not in href and "/sport/" not in href:
                continue
                
            seen_urls.add(href)
            
            if not href.startswith("http"):
                href = "https://www.bbc.co.uk" + href if href.startswith("/") else "https://www.bbc.co.uk/" + href
            
            headline = a.get_text(strip=True)
            if not headline or len(headline) < 10:
                continue
            
            results.append(f"BBC News: {headline}")
            print(f"   ✅ Found: {headline[:60]}...")
            
            if len(results) >= max_results:
                break
        
        if not results:
            print(f"   ⚠️  No BBC results found")
        return results
        
    except Exception as e:
        print(f"   ❌ BBC error: {e}")
        return []

def generate_search_queries(draft: str) -> List[str]:
    """Use LLM to generate optimal search queries."""
    print(f"\n   🤖 Generating search queries...")
    prompt = f"""Generate 2-3 search queries to verify this claim:

"{draft}"

Respond with JSON: {{"queries": ["query1", "query2"]}}"""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed and "queries" in parsed:
        queries = parsed["queries"]
        print(f"   ✅ Generated queries: {queries}")
        return queries
    else:
        fallback = [" ".join(draft.split()[:5])]
        print(f"   ⚠️  Using fallback query: {fallback}")
        return fallback

# -----------------------------
# NLI SCORING
# -----------------------------

def nli_score(premise: str, hypothesis: str):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    return {
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2]),
    }

# -----------------------------
# HELPERS
# -----------------------------

def extract_json_from_text(text: str) -> Optional[dict]:
    if text is None:
        return None
    
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass
    
    pattern = r"```(?:json)?\s*({.*?})\s*```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    brace_pattern = r"(\{(?:[^{}]|\{.*?\})*\})"
    m = re.search(brace_pattern, text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    
    return None

# -----------------------------
# STATE DEFINITION
# -----------------------------

class BullshitState(TypedDict):
    user_id: str
    draft: str
    user_contexts: List[str]
    external_contexts: List[str]
    combined_contexts: List[str]
    nli_contradictions: List[str]
    nli_scores: List[dict]
    llm_response: object
    tool_calls_made: List[str]

# -----------------------------
# OPTION 1: EVIDENCE-FIRST
# -----------------------------

def evidence_first_retrieve(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("STEP 1: RETRIEVING CONTEXT")
    print("="*70)
    
    user_id = state["user_id"]
    draft = state["draft"]
    
    print("\n📚 Retrieving user's previous posts...")
    user_contexts = retrieve_similar(user_id, draft, top_k=3)
    print(f"✅ Found {len(user_contexts)} relevant previous posts:\n")
    for i, ctx in enumerate(user_contexts, 1):
        print(f"   {i}. \"{ctx}\"")
    
    print("\n🔎 Generating and executing search queries...")
    queries = generate_search_queries(draft)
    
    external_contexts = []
    print("\n📰 EXTERNAL SOURCES:")
    print("-" * 70)
    
    for query in queries[:2]:
        wiki = fetch_wikipedia(query)
        if wiki:
            external_contexts.append(f"Wikipedia({query}): {wiki}")
            print(f"\n📖 Wikipedia - {query}:")
            print(f"   {wiki[:300]}...")
            print()
    
    if queries:
        bbc = fetch_bbc(queries[0], max_results=2)
        for item in bbc:
            external_contexts.append(item)
            print(f"\n📰 {item}")
            print()
    
    print(f"✅ Gathered {len(external_contexts)} external sources")
    
    state["user_contexts"] = user_contexts
    state["external_contexts"] = external_contexts
    state["combined_contexts"] = user_contexts + external_contexts
    state["tool_calls_made"] = ["retrieve_user_history", "generate_queries", "fetch_wikipedia", "fetch_bbc"]
    return state

def evidence_first_nli(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("STEP 2: CHECKING FOR CONTRADICTIONS (NLI)")
    print("="*70)
    
    contradictions = []
    print(f"\n🔬 Running NLI on {len(state['combined_contexts'])} context items...")
    
    for ctx in state["combined_contexts"]:
        if not ctx.strip():
            continue
        sc = nli_score(ctx, state["draft"])
        if sc["contradiction"] > 0.6:
            contradictions.append(ctx)
            print(f"   ⚠️  CONTRADICTION DETECTED (score: {sc['contradiction']:.2f})")
            print(f"       Context: {ctx[:80]}...")
    
    if not contradictions:
        print("   ✅ No contradictions detected")
    
    state["nli_contradictions"] = contradictions
    state["tool_calls_made"].append("run_nli")
    return state

def evidence_first_assess(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("STEP 3: FINAL ASSESSMENT")
    print("="*70)
    
    print("\n🤖 LLM analyzing all gathered evidence...")
    
    ctx_for_prompt = "\n".join(f"- {c[:300]}" for c in state["combined_contexts"][:10])
    contradictions_list = "\n".join(f"- {c[:200]}" for c in state.get('nli_contradictions', [])[:5])
    
    prompt = f"""Analyze this tweet draft:

"{state['draft']}"

Context: {ctx_for_prompt}
Contradictions: {contradictions_list if contradictions_list else "None"}

Classify as: bullshit, not_bullshit, or contextually_ambiguous

Respond with JSON:
{{
  "classification": "...",
  "confidence": "high/medium/low",
  "reasoning": "...",
  "suggested_revision": "..."
}}"""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed:
        state["llm_response"] = parsed
    else:
        state["llm_response"] = {"classification": "unknown", "confidence": "low", "reasoning": raw[:200], "suggested_revision": ""}
    
    return state

def build_evidence_first():
    graph = StateGraph(BullshitState)
    graph.add_node("retrieve", evidence_first_retrieve)
    graph.add_node("nli", evidence_first_nli)
    graph.add_node("assess", evidence_first_assess)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "nli")
    graph.add_edge("nli", "assess")
    graph.add_edge("assess", END)
    return graph.compile()

# -----------------------------
# OPTION 2: REACT
# -----------------------------

def react_agent(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("REACT: AGENT-DRIVEN VERIFICATION")
    print("="*70)
    
    user_id = state["user_id"]
    draft = state["draft"]
    accumulated_context = []
    tool_log = []
    
    print("\n📚 Retrieving user history...")
    user_contexts = retrieve_similar(user_id, draft, top_k=3)
    print(f"✅ Found {len(user_contexts)} relevant previous posts:\n")
    for i, ctx in enumerate(user_contexts, 1):
        print(f"   {i}. \"{ctx}\"")
    
    accumulated_context.extend(user_contexts)
    tool_log.append("retrieve_user_history")
    
    max_rounds = 3
    for round_idx in range(max_rounds):
        print(f"\n{'─'*70}")
        print(f"ROUND {round_idx + 1}: Agent deciding next action...")
        print(f"{'─'*70}")
        
        prompt = f"""You are fact-checking: "{draft}"

Context so far: {accumulated_context if accumulated_context else "[None]"}

Available tools: fetch_wikipedia, fetch_bbc, run_nli, assess (finish)

What should you do next? Respond with JSON:
{{"action": "fetch_wikipedia", "query": "..."}} or {{"action": "assess"}}"""
        
        resp = chat.invoke([{"role": "user", "content": prompt}])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = extract_json_from_text(raw)
        
        if not parsed or "action" not in parsed:
            print("   ⚠️  Could not parse decision, stopping")
            break
        
        action = parsed.get("action")
        print(f"   🤖 Agent chose: {action}")
        
        if action == "fetch_wikipedia":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            wiki = fetch_wikipedia(query)
            if wiki:
                accumulated_context.append(f"Wikipedia({query}): {wiki}")
                tool_log.append(f"fetch_wikipedia({query})")
                print(f"\n   📖 Wikipedia Result:")
                print(f"   {wiki[:300]}...\n")
        
        elif action == "fetch_bbc":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            bbc = fetch_bbc(query, max_results=2)
            for item in bbc:
                accumulated_context.append(item)
                print(f"\n   📰 {item}\n")
            if bbc:
                tool_log.append(f"fetch_bbc({query})")
        
        elif action == "run_nli":
            print("   🔬 Running NLI...")
            contradictions = []
            for ctx in accumulated_context:
                if not ctx.strip():
                    continue
                sc = nli_score(ctx, draft)
                if sc["contradiction"] > 0.6:
                    contradictions.append(ctx)
                    print(f"      ⚠️  Contradiction found (score: {sc['contradiction']:.2f})")
                    print(f"      Context: \"{ctx[:100]}...\"")
            state["nli_contradictions"] = contradictions
            tool_log.append("run_nli")
        
        elif action == "assess":
            print("   ✅ Agent decided to finish")
            break
        
        else:
            print(f"   ⚠️  Unknown action, stopping")
            break
    
    state["user_contexts"] = user_contexts
    state["external_contexts"] = accumulated_context[len(user_contexts):]
    state["combined_contexts"] = accumulated_context
    state["tool_calls_made"] = tool_log
    return state

def react_assess(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    
    print("\n🤖 LLM making final judgment...")
    
    ctx_for_prompt = "\n".join(f"- {c[:300]}" for c in state["combined_contexts"][:10])
    
    prompt = f"""Analyze: "{state['draft']}"

Context: {ctx_for_prompt}

Classify as: bullshit, not_bullshit, or contextually_ambiguous

JSON response:
{{
  "classification": "...",
  "confidence": "...",
  "reasoning": "...",
  "suggested_revision": "..."
}}"""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed:
        state["llm_response"] = parsed
    else:
        state["llm_response"] = {"classification": "unknown", "confidence": "low", "reasoning": raw[:200], "suggested_revision": ""}
    
    return state

def build_react():
    graph = StateGraph(BullshitState)
    graph.add_node("react", react_agent)
    graph.add_node("assess", react_assess)
    graph.add_edge(START, "react")
    graph.add_edge("react", "assess")
    graph.add_edge("assess", END)
    return graph.compile()

# -----------------------------
# OPTION 3: INFERENCE-FIRST
# -----------------------------

def inference_first(state: BullshitState) -> BullshitState:
    print("\n" + "="*70)
    print("INFERENCE-FIRST: DIRECT LLM JUDGMENT (NO TOOLS)")
    print("="*70)
    
    print("\n🤖 LLM making judgment based solely on internal knowledge...")
    print("   (No retrieval, no search, no NLI)")
    
    draft = state["draft"]
    
    prompt = f"""Analyze this tweet using only your internal knowledge:

"{draft}"

No external tools. Classify as: bullshit, not_bullshit, or contextually_ambiguous

JSON response:
{{
  "classification": "...",
  "confidence": "...",
  "reasoning": "...",
  "suggested_revision": "..."
}}"""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed:
        state["llm_response"] = parsed
    else:
        state["llm_response"] = {"classification": "unknown", "confidence": "low", "reasoning": raw[:200], "suggested_revision": ""}
    
    state["user_contexts"] = []
    state["external_contexts"] = []
    state["combined_contexts"] = []
    state["nli_contradictions"] = []
    state["tool_calls_made"] = ["direct_inference_only"]
    
    return state

def build_inference_first():
    graph = StateGraph(BullshitState)
    graph.add_node("assess", inference_first)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    return graph.compile()

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

def display_results(result: Dict[str, Any], system_name: str):
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    llm_resp = result.get("llm_response", {})
    
    classification = llm_resp.get("classification", "unknown").upper()
    confidence = llm_resp.get("confidence", "unknown").upper()
    reasoning = llm_resp.get("reasoning", "No reasoning provided")
    suggested = llm_resp.get("suggested_revision", "")
    
    # Color code classification
    if "bullshit" in classification.lower():
        class_display = f"🚨 {classification}"
    elif "not_bullshit" in classification.lower():
        class_display = f"✅ {classification}"
    else:
        class_display = f"⚠️  {classification}"
    
    print(f"\n📊 CLASSIFICATION: {class_display}")
    print(f"📈 CONFIDENCE: {confidence}")
    print(f"\n💭 REASONING:")
    print(f"   {reasoning}")
    
    if suggested:
        print(f"\n✏️  SUGGESTED REVISION:")
        print(f"   {suggested}")
    
    # Show user context that was retrieved
    user_contexts = result.get("user_contexts", [])
    if user_contexts:
        print(f"\n👤 USER'S PREVIOUS POSTS REFERENCED ({len(user_contexts)}):")
        for i, ctx in enumerate(user_contexts, 1):
            print(f"   {i}. \"{ctx}\"")
    
    # Show external sources
    external_contexts = result.get("external_contexts", [])
    if external_contexts:
        print(f"\n🌐 EXTERNAL SOURCES CONSULTED ({len(external_contexts)}):")
        for i, ctx in enumerate(external_contexts[:5], 1):
            # Extract just the source name and first part
            if ctx.startswith("Wikipedia("):
                parts = ctx.split("): ", 1)
                if len(parts) == 2:
                    source = parts[0] + ")"
                    content = parts[1][:200]
                    print(f"   {i}. {source}")
                    print(f"      \"{content}...\"")
                else:
                    print(f"   {i}. {ctx[:250]}...")
            elif ctx.startswith("BBC News:"):
                print(f"   {i}. {ctx}")
            else:
                print(f"   {i}. {ctx[:250]}...")
    
    # Show contradictions
    contradictions = result.get("nli_contradictions", [])
    if contradictions:
        print(f"\n⚠️  CONTRADICTIONS DETECTED ({len(contradictions)}):")
        for i, c in enumerate(contradictions[:3], 1):
            print(f"   {i}. \"{c[:150]}...\"")
    
    tools = result.get("tool_calls_made", [])
    print(f"\n🔧 TOOLS USED: {', '.join(tools)}")
    print(f"📦 TOTAL CONTEXT ITEMS: {len(result.get('combined_contexts', []))}")
    print(f"   - User history: {len(user_contexts)}")
    print(f"   - External sources: {len(external_contexts)}")

# -----------------------------
# MAIN DEMO
# -----------------------------

def run_demo():
    print("\n" + "🎯"*35)
    print("BULLSHIT DETECTOR - INTERACTIVE DEMO")
    print("🎯"*35)
    
    print("\nYour previous posts (context):")
    for i, post in enumerate(DEMO_USER_HISTORY[:5], 1):
        print(f"  {i}. {post}")
    print(f"  ... and {len(DEMO_USER_HISTORY) - 5} more posts\n")
    
    print("="*70)
    print("CHOOSE A VERIFICATION SYSTEM:")
    print("="*70)
    print("\n1️⃣  EVIDENCE-FIRST (Predefined Pipeline)")
    print("   - Fixed steps: retrieve → NLI → assess")
    print("   - Comprehensive, always checks everything")
    print("   - Predictable workflow\n")
    
    print("2️⃣  REACT (Agent-Driven)")
    print("   - Agent decides which tools to use")
    print("   - Adaptive, can skip unnecessary steps")
    print("   - More efficient\n")
    
    print("3️⃣  INFERENCE-FIRST (Baseline)")
    print("   - No tools, just LLM knowledge")
    print("   - Fast but limited")
    print("   - No external verification\n")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        system_name = "Evidence-First"
        graph = build_evidence_first()
    elif choice == "2":
        system_name = "ReACT"
        graph = build_react()
    elif choice == "3":
        system_name = "Inference-First"
        graph = build_inference_first()
    else:
        print("Invalid choice. Defaulting to Evidence-First.")
        system_name = "Evidence-First"
        graph = build_evidence_first()
    
    print(f"\n✅ Selected: {system_name}\n")
    
    draft = input("Enter a tweet/claim to verify: ").strip()
    
    if not draft:
        draft = "Climate change is a hoax invented by scientists to get funding"
        print(f"Using example: {draft}")
    
    print("\n" + "🚀"*35)
    print(f"RUNNING {system_name.upper()}")
    print("🚀"*35)
    
    state0 = {
        "user_id": DEMO_USER_ID,
        "draft": draft,
        "user_contexts": [],
        "external_contexts": [],
        "combined_contexts": [],
        "nli_contradictions": [],
        "nli_scores": [],
        "llm_response": {},
        "tool_calls_made": []
    }
    
    start_time = time.time()
    result = graph.invoke(state0)
    elapsed = time.time() - start_time
    
    display_results(result, system_name)
    
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print("\n" + "="*70)

if __name__ == "__main__":
    print("\n✅ Models loaded successfully!\n")
    run_demo()
    
    while True:
        again = input("\n\nRun another demo? (y/n): ").strip().lower()
        if again == 'y':
            run_demo()
        else:
            print("\n👋 Thanks for trying the Bullshit Detector!\n")
            break