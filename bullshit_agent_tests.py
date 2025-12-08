# ==========================================
# bullshit_agent_langgraph.py  (with all ablations + evaluation)
# ==========================================

from typing import TypedDict, Annotated, List, Optional, Dict, Any
import faiss
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import json
import re
import os
from pathlib import Path
from collections import defaultdict

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
    
    # Skip if no texts to add
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

# -----------------------------
# 3. EXTERNAL CONTEXT TOOLS
# -----------------------------

def fetch_wikipedia(query: str) -> str:
    """Fetch Wikipedia summary for a query."""
    print(f"in fetch wiki. Query: {query}")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            return extract
        else:
            # Try alternative: use Wikipedia's search API
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            try:
                search_r = requests.get(search_url, headers=headers, timeout=10)
                if search_r.status_code == 200:
                    search_data = search_r.json()
                    results = search_data.get("query", {}).get("search", [])
                    if results:
                        snippet = results[0].get("snippet", "")
                        # Remove HTML tags
                        snippet = re.sub('<.*?>', '', snippet)
                        return snippet
            except Exception:
                pass
    except Exception as e:
        print(f"Wikipedia fetch error: {e}")
    return ""

def fetch_bbc(query: str, max_results: int = 2) -> List[str]:
    """Fetch BBC news articles for a query."""
    print(f"in fetch bbc. Query: {query}")
    url = f"https://www.bbc.co.uk/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code != 200:
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        
        # Try multiple selectors for BBC's layout
        articles = (soup.select("article a[href*='/news/']") or 
                   soup.select("a[href*='/news/']") or
                   soup.select("div.ssrcss-1f3bvyz-Stack a"))
        
        seen_urls = set()
        for a in articles[:max_results * 3]:  # Get more candidates
            href = a.get("href", "")
            if not href or href in seen_urls:
                continue
            
            # Make sure it's a news article URL
            if "/news/" not in href and "/sport/" not in href:
                continue
                
            seen_urls.add(href)
            
            if not href.startswith("http"):
                href = "https://www.bbc.co.uk" + href if href.startswith("/") else "https://www.bbc.co.uk/" + href
            
            headline = a.get_text(strip=True)
            if not headline or len(headline) < 10:
                continue
            
            # Return just headline to avoid too many requests
            results.append(f"BBC News: {headline}")
            
            if len(results) >= max_results:
                break
        
        return results
        
    except Exception as e:
        print(f"BBC fetch error: {e}")
        return []

def generate_search_queries(draft: str) -> List[str]:
    """Use LLM to generate optimal search queries for a draft claim."""
    print("in generate_search_queries")
    prompt = f"""You are a research assistant helping to verify a claim.

Claim: "{draft}"

Generate 2-3 specific search queries that would help verify or refute this claim.
Focus on:
- Key entities, events, or concepts
- Verifiable facts or statistics
- Recent developments if time-sensitive

Respond with JSON:
{{"queries": ["query1", "query2", "query3"]}}

Respond ONLY with JSON."""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed and "queries" in parsed:
        return parsed["queries"]
    else:
        # Fallback: extract first 5 words
        return [" ".join(draft.split()[:5])]

# -----------------------------
# 4. NLI SCORING
# -----------------------------

def nli_score(premise: str, hypothesis: str):
    print("in nli score")
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
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
    """Extract and parse a JSON object from text."""
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
        block = match.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass

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
    llm_response: object
    tool_calls_made: Annotated[List[str], "log of tools used"]

# ============================================================================
# 6. ABLATION 1: EVIDENCE-FIRST (Predefined Pipeline)
# ============================================================================
# DESIGN PRINCIPLE: Fixed sequence of verification steps
# 
# WORKFLOW:
#   1. retrieve_node → Fetch user history + generate queries + search external sources
#   2. nli_node → Run NLI contradiction detection on all gathered context
#   3. assess_node → LLM makes final judgment with all context
# 
# KEY CHARACTERISTICS:
#   - No dynamic decision-making by the agent
#   - Always executes all steps in the same order
#   - Uses LLM for query generation but not for workflow control
#   - Predictable, deterministic pipeline
# ============================================================================

def evidence_first_retrieve_node(state: BullshitState) -> BullshitState:
    """
    STEP 1: Retrieve all context upfront (predefined).
    - Fetch user history from vector DB
    - Generate search queries using LLM
    - Execute Wikipedia and BBC searches for generated queries
    """
    user_id = state["user_id"]
    draft = state["draft"]
    
    print("[Evidence-First] Step 1: Retrieving user history...")
    user_contexts = retrieve_similar(user_id, draft, top_k=5)
    state["tool_calls_made"].append("retrieve_user_history")
    
    print("[Evidence-First] Step 2: Generating search queries...")
    queries = generate_search_queries(draft)
    
    print("[Evidence-First] Step 3: Executing searches...")
    external_contexts = []
    
    # Search Wikipedia for first 2 queries
    for query in queries[:2]:
        wiki = fetch_wikipedia(query)
        if wiki:
            external_contexts.append(f"Wikipedia({query}): {wiki}")
            state["tool_calls_made"].append(f"fetch_wikipedia({query})")
    
    # Search BBC for primary query
    if queries:
        bbc = fetch_bbc(queries[0], max_results=2)
        external_contexts.extend(bbc)
        if bbc:
            state["tool_calls_made"].append(f"fetch_bbc({queries[0]})")
    
    state["user_contexts"] = user_contexts
    state["external_contexts"] = external_contexts
    state["combined_contexts"] = user_contexts + external_contexts
    return state

def nli_node(state: BullshitState) -> BullshitState:
    """
    STEP 2: Run NLI contradiction detection (always executed).
    Checks draft against all gathered context for contradictions.
    """
    print("[Evidence-First] Step 4: Running NLI contradiction detection...")
    scores = []
    contradictions = []
    for ctx in state["combined_contexts"]:
        if not ctx.strip():
            continue
        sc = nli_score(ctx, state["draft"])
        scores.append(sc)
        if sc["contradiction"] > 0.6:
            contradictions.append(ctx)
    
    state["nli_scores"] = scores
    state["nli_contradictions"] = contradictions
    state["tool_calls_made"].append("run_nli")
    return state

def llm_assess_node(state: BullshitState) -> BullshitState:
    """
    STEP 3: Final LLM assessment (always executed).
    Synthesizes all context and NLI results into comprehensive assessment.
    """
    print("[Evidence-First] Step 5: Final LLM assessment...")
    ctx_for_prompt = "\n".join(f"- {c[:300]}" for c in state["combined_contexts"][:10])
    contradictions_list = "\n".join(f"- {c[:200]}" for c in state.get('nli_contradictions', [])[:5])
    
    prompt = f"""You are a fact-checking assistant analyzing a tweet draft.

Draft Tweet:
\"\"\"{state['draft']}\"\"\"

User's Historical Context (previous posts):
{ctx_for_prompt if ctx_for_prompt else "[No user history available]"}

External Sources:
{ctx_for_prompt}

NLI Contradictions Found: {len(state.get('nli_contradictions', []))}
{contradictions_list if contradictions_list else ""}

Analyze the draft and provide a comprehensive assessment:

1. Classify as: "bullshit", "not_bullshit", or "contextually_ambiguous"
   - bullshit: False, misleading, or overconfident claims
   - not_bullshit: Accurate, grounded, well-supported statements
   - contextually_ambiguous: Unclear, vague, or context-dependent

2. Identify internal contradictions (with user's previous posts)
3. Identify external contradictions (with factual sources - provide references)
4. Assess confidence in your classification
5. Provide reasoning
6. Suggest a more grounded revision if needed

Respond with JSON:
{{
  "tweet_summary": "brief summary of the draft tweet",
  "context_summary": "brief summary of gathered context",
  "classification": "bullshit/not_bullshit/contextually_ambiguous",
  "confidence": "high/medium/low",
  "internal_contradictions": ["contradiction with user's post: ...", ...],
  "external_contradictions": [
    {{"claim": "...", "contradiction": "...", "source": "Wikipedia: ... or BBC: ..."}}
  ],
  "reasoning": "detailed explanation of classification",
  "suggested_revision": "revised tweet if needed, or empty string if not needed"
}}
"""
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed is None:
        state["llm_response"] = {
            "tweet_summary": raw[:200],
            "context_summary": "Parse failed",
            "classification": "unknown",
            "confidence": "low",
            "internal_contradictions": [],
            "external_contradictions": [],
            "reasoning": "Failed to parse response",
            "suggested_revision": ""
        }
    else:
        state["llm_response"] = parsed
    
    state["tool_calls_made"].append("llm_assess")
    return state

def build_evidence_first_graph():
    """Build the Evidence-First graph with fixed pipeline."""
    graph = StateGraph(BullshitState)
    graph.add_node("retrieve", evidence_first_retrieve_node)
    graph.add_node("nli", nli_node)
    graph.add_node("assess", llm_assess_node)
    
    # Fixed pipeline: always this exact sequence
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "nli")
    graph.add_edge("nli", "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# ============================================================================
# 7. ABLATION 2: REACT (Agent-Driven Tool Selection)
# ============================================================================
# DESIGN PRINCIPLE: Agent dynamically decides which tools to use and when
# 
# WORKFLOW:
#   1. react_node → Agent iteratively decides: fetch_wikipedia | fetch_bbc | 
#                   run_nli | assess (finish)
#   2. assess_node → Final LLM assessment after agent finishes gathering
# 
# KEY CHARACTERISTICS:
#   - Agent has full autonomy over tool selection
#   - Can skip tools it deems unnecessary
#   - Can stop early if confident in judgment
#   - Workflow adapts to the specific claim being verified
#   - More efficient but less predictable than Evidence-First
# ============================================================================

def react_agent_node(state: BullshitState) -> BullshitState:
    """
    REACT LOOP: Agent decides which verification steps to take.
    
    Available actions:
    - fetch_wikipedia: Search Wikipedia for a topic
    - fetch_bbc: Search BBC news for a topic
    - run_nli: Check for contradictions
    - assess: Finish and make judgment
    
    Agent continues until it calls 'assess' or reaches max_rounds.
    """
    user_id = state["user_id"]
    draft = state["draft"]
    accumulated_context: List[str] = []
    tool_log = state.get("tool_calls_made", [])
    failed_attempts = []  # Track failed attempts to avoid loops
    
    # Always start with user history (baseline context)
    print("[ReACT] Retrieving user history...")
    user_contexts = retrieve_similar(user_id, draft, top_k=5)
    accumulated_context.extend(user_contexts)
    tool_log.append("retrieve_user_history")
    
    max_rounds = 5
    for round_idx in range(max_rounds):
        print(f"\n[ReACT] Round {round_idx + 1}: Agent deciding next action...")
        
        # Ask agent what to do next
        prompt = f"""You are a fact-checking agent verifying a tweet draft.

Draft:
\"\"\"{draft}\"\"\"

Current context gathered:
{accumulated_context if accumulated_context else "[None yet]"}

Tools used so far: {tool_log}
Failed attempts: {failed_attempts if failed_attempts else "[None]"}

Available tools:
- fetch_wikipedia: Get Wikipedia summary for a topic
- fetch_bbc: Get recent news articles
- run_nli: Check for contradictions with existing context
- assess: Make final judgment and finish

IMPORTANT: If a tool failed, try a different tool or query. Don't repeat failures.

Decide what to do next. Respond with JSON:

To use a tool:
{{"action": "fetch_wikipedia", "query": "topic"}}
{{"action": "fetch_bbc", "query": "topic"}}
{{"action": "run_nli"}}

To finish:
{{"action": "assess"}}

Respond ONLY with JSON."""
        
        resp = chat.invoke([{"role": "user", "content": prompt}])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = extract_json_from_text(raw)
        
        if not parsed or "action" not in parsed:
            print("[ReACT] Could not parse agent decision. Stopping.")
            break
        
        action = parsed.get("action")
        print(f"[ReACT] Agent chose action: {action}")
        
        # Track attempts to prevent loops
        attempt_key = None
        if action in ["fetch_wikipedia", "fetch_bbc"]:
            query = parsed.get("query", " ".join(draft.split()[:3]))
            attempt_key = f"{action}:{query}"
            if attempt_key in failed_attempts:
                print(f"[ReACT] This attempt already failed. Agent should try something else.")
                if round_idx < max_rounds - 1:
                    continue
                else:
                    break
        
        # Execute the chosen tool
        if action == "fetch_wikipedia":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            wiki = fetch_wikipedia(query)
            if wiki:
                accumulated_context.append(f"Wikipedia({query}): {wiki}")
                tool_log.append(f"fetch_wikipedia({query})")
            else:
                failed_attempts.append(attempt_key)
        
        elif action == "fetch_bbc":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            bbc_items = fetch_bbc(query, max_results=2)
            if bbc_items:
                accumulated_context.extend(bbc_items)
                tool_log.append(f"fetch_bbc({query})")
            else:
                failed_attempts.append(attempt_key)
        
        elif action == "run_nli":
            print("[ReACT] Running NLI...")
            nli_contradictions = []
            nli_scores = []
            for ctx in accumulated_context:
                if not ctx.strip():
                    continue
                sc = nli_score(ctx, draft)
                nli_scores.append(sc)
                if sc["contradiction"] > 0.6:
                    nli_contradictions.append(ctx)
            
            state["nli_contradictions"] = nli_contradictions
            state["nli_scores"] = nli_scores
            tool_log.append("run_nli")
            
            # Add NLI results to context for agent's next decision
            accumulated_context.append(f"NLI Analysis: Found {len(nli_contradictions)} contradictions")
        
        elif action == "assess":
            print("[ReACT] Agent decided to assess and finish.")
            break
        
        else:
            print(f"[ReACT] Unknown action: {action}. Stopping.")
            break
        
        # Safety: stop if too many failures with no success
        if len(failed_attempts) >= 3 and len(accumulated_context) <= len(user_contexts):
            print("[ReACT] Multiple failed attempts. Stopping.")
            break
    
    # Store final state
    state["user_contexts"] = user_contexts
    state["external_contexts"] = accumulated_context[len(user_contexts):]
    state["combined_contexts"] = accumulated_context
    state["tool_calls_made"] = tool_log
    
    return state

def react_assess_node(state: BullshitState) -> BullshitState:
    """Final assessment after ReACT tool selection."""
    print("[ReACT] Making final assessment...")
    ctx_for_prompt = "\n".join(f"- {c[:300]}" for c in state["combined_contexts"][:10])
    contradictions_list = "\n".join(f"- {c[:200]}" for c in state.get('nli_contradictions', [])[:5])
    
    prompt = f"""You are a fact-checking assistant analyzing a tweet draft.

Draft Tweet:
\"\"\"{state['draft']}\"\"\"

User's Historical Context (previous posts):
{ctx_for_prompt if ctx_for_prompt else "[No user history available]"}

External Sources:
{ctx_for_prompt}

Tools Used: {state.get('tool_calls_made', [])}

NLI Contradictions Found: {len(state.get('nli_contradictions', []))}
{contradictions_list if contradictions_list else ""}

Analyze the draft and provide a comprehensive assessment:

1. Classify as: "bullshit", "not_bullshit", or "contextually_ambiguous"
   - bullshit: False, misleading, or overconfident claims
   - not_bullshit: Accurate, grounded, well-supported statements
   - contextually_ambiguous: Unclear, vague, or context-dependent

2. Identify internal contradictions (with user's previous posts)
3. Identify external contradictions (with factual sources - provide references)
4. Assess confidence in your classification
5. Provide reasoning
6. Suggest a more grounded revision if needed

Respond with JSON:
{{
  "tweet_summary": "brief summary of the draft tweet",
  "context_summary": "brief summary of gathered context",
  "classification": "bullshit/not_bullshit/contextually_ambiguous",
  "confidence": "0.00-1.00",
  "internal_contradictions": ["contradiction with user's post: ...", ...],
  "external_contradictions": [
    {{"claim": "...", "contradiction": "...", "source": "Wikipedia: ... or BBC: ..."}}
  ],
  "reasoning": "detailed explanation of classification",
  "suggested_revision": "revised tweet if needed, or empty string if not needed"
}}
"""
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed is None:
        state["llm_response"] = {
            "tweet_summary": raw[:200],
            "context_summary": "Parse failed",
            "classification": "unknown",
            "confidence": "low",
            "internal_contradictions": [],
            "external_contradictions": [],
            "reasoning": "Failed to parse response",
            "suggested_revision": ""
        }
    else:
        state["llm_response"] = parsed
    
    state["tool_calls_made"].append("llm_assess")
    return state

def build_react_graph():
    """Build the ReACT graph with agent-driven workflow."""
    graph = StateGraph(BullshitState)
    graph.add_node("react", react_agent_node)
    graph.add_node("assess", react_assess_node)
    
    # Agent-driven flow: react node makes decisions, then final assessment
    graph.add_edge(START, "react")
    graph.add_edge("react", "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# ============================================================================
# 8. ABLATION 3: INFERENCE-FIRST (Non-Agentic Baseline)
# ============================================================================
# DESIGN PRINCIPLE: No tools, no retrieval - pure LLM judgment
# 
# WORKFLOW:
#   1. assess_node → LLM makes judgment based solely on internal knowledge
# 
# KEY CHARACTERISTICS:
#   - No external verification
#   - No user history retrieval
#   - No NLI checking
#   - Baseline to measure value of tools and retrieval
#   - Fast but potentially unreliable
# ============================================================================

def inference_first_node(state: BullshitState) -> BullshitState:
    """Direct LLM judgment with no retrieval or tools."""
    draft = state["draft"]
    
    print("[Inference-First] Making direct judgment with no tools...")
    
    prompt = f"""You are a fact-checking assistant analyzing a tweet draft based solely on your internal knowledge.

Draft Tweet:
\"\"\"{draft}\"\"\"

Without retrieving any external information or user history, assess:

1. Classify as: "bullshit", "not_bullshit", or "contextually_ambiguous"
   - bullshit: False, misleading, or overconfident claims
   - not_bullshit: Accurate, grounded, well-supported statements
   - contextually_ambiguous: Unclear, vague, or context-dependent

2. Assess confidence in your classification
3. Provide reasoning based on general knowledge
4. Suggest a more grounded revision if needed

Respond with JSON:
{{
  "tweet_summary": "brief summary of the draft tweet",
  "context_summary": "no external context used",
  "classification": "bullshit/not_bullshit/contextually_ambiguous",
  "confidence": "high/medium/low",
  "internal_contradictions": [],
  "external_contradictions": [],
  "reasoning": "detailed explanation based on internal knowledge",
  "suggested_revision": "revised tweet if needed, or empty string if not needed"
}}
"""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if parsed is None:
        state["llm_response"] = {
            "tweet_summary": raw[:200],
            "context_summary": "No external context",
            "classification": "unknown",
            "confidence": "low",
            "internal_contradictions": [],
            "external_contradictions": [],
            "reasoning": "Failed to parse response",
            "suggested_revision": ""
        }
    else:
        state["llm_response"] = parsed
    
    # No retrieval, so contexts are empty
    state["user_contexts"] = []
    state["external_contexts"] = []
    state["combined_contexts"] = []
    state["nli_contradictions"] = []
    state["nli_scores"] = []
    state["tool_calls_made"] = ["direct_inference_only"]
    
    return state

def build_inference_first_graph():
    """Build the Inference-First graph (baseline with no tools)."""
    graph = StateGraph(BullshitState)
    graph.add_node("assess", inference_first_node)
    
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# -----------------------------
# 9. DATA LOADING & PREPROCESSING
# -----------------------------

import pandas as pd

def prepare_timeline_data(csv_path: str) -> Dict[str, Any]:
    """
    Prepare timeline data from CSV.
    - Rows WITH labels 'bullshit', 'not bullshit', or 'contextually ambiguous' are TEST CASES
    - Rows WITH label 'context' are HISTORICAL CONTEXT
    - Rows WITHOUT labels (empty/NaN) are also HISTORICAL CONTEXT
    """
    df = pd.read_csv(csv_path)
    
    # Remove rows with empty text
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']
    
    if len(df) == 0:
        print(f"Warning: No valid data in {csv_path}")
        return None
    
    # Sort by date if available
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Standardize label names
    label_map = {
        'bullshit': 'bullshit',
        'not bullshit': 'not_bullshit',
        'contextually ambiguous': 'contextually_ambiguous',
        'contextually-ambiguous': 'contextually_ambiguous',
        'context': 'context',  # Add context as valid label
    }
    
    # Create a copy and standardize labels
    df['LABEL_CLEAN'] = df['LABEL'].fillna('').str.lower().str.strip()
    df['LABEL_CLEAN'] = df['LABEL_CLEAN'].map(lambda x: label_map.get(x, 'context' if x == '' else None))
    
    # Separate into test cases and context
    # Test cases: bullshit, not_bullshit, contextually_ambiguous
    test_labels = ['bullshit', 'not_bullshit', 'contextually_ambiguous']
    test_mask = df['LABEL_CLEAN'].isin(test_labels)
    test_df = df[test_mask].copy()
    
    # Context: rows labeled 'context' OR rows with empty/NaN labels
    context_mask = (df['LABEL_CLEAN'] == 'context') | (df['LABEL_CLEAN'].isna()) | (df['LABEL_CLEAN'] == '')
    context_df = df[context_mask].copy()
    
    if len(test_df) == 0:
        print(f"Warning: No labeled test cases found in {csv_path}")
        return None
    
    # Build timeline structure
    user_id = df['profile'].iloc[0] if 'profile' in df.columns else Path(csv_path).stem
    
    timeline_data = {
        "user_id": user_id,
        "source_file": str(csv_path),
        "history": context_df['text'].tolist(),
        "test_cases": []
    }
    
    # Create test cases from labeled rows
    for _, row in test_df.iterrows():
        test_case = {
            "draft": row['text'],
            "label": row['LABEL_CLEAN'],
            "date": row.get('date', ''),
            "likes": int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            "retweets": int(row.get('retweets', 0)) if pd.notna(row.get('retweets')) else 0,
            "link": row.get('link', ''),
            "has_contradiction": row['LABEL_CLEAN'] == 'bullshit'
        }
        timeline_data["test_cases"].append(test_case)
    
    print(f"Loaded {user_id}: {len(timeline_data['history'])} context tweets, {len(timeline_data['test_cases'])} test cases")
    
    return timeline_data

def load_all_timelines(data_dir: str = "data") -> List[Dict[str, Any]]:
    """Load all CSV files from data directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found")
        return []
    
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in '{data_dir}'")
        return []
    
    timelines = []
    for csv_file in csv_files:
        try:
            timeline = prepare_timeline_data(csv_file)
            if timeline:
                timelines.append(timeline)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return timelines

def evaluate_system(system_name: str, graph, timeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a system on a timeline."""
    user_id = timeline_data.get("user_id", "test_user")
    history = timeline_data.get("history", [])
    test_cases = timeline_data.get("test_cases", [])
    
    print(f"\n{'='*70}")
    print(f"User: {user_id}")
    print(f"Context tweets: {len(history)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'='*70}")
    
    # Load user history into vector DB
    if user_id not in user_data:
        user_data[user_id] = {"texts": history}
        add_user_texts(user_id, history)
    
    results = []
    metrics = {
        "correct": 0,
        "total": 0,
        "context_used": 0,
        "contradiction_detected": 0,
        "contradiction_expected": 0,
        "avg_tool_calls": 0,
        "by_label": {
            "bullshit": {"correct": 0, "total": 0},
            "not_bullshit": {"correct": 0, "total": 0},
            "contextually_ambiguous": {"correct": 0, "total": 0}
        }
    }
    
    for idx, test_case in enumerate(test_cases, 1):
        draft = test_case["draft"]
        expected_label = test_case["label"]
        has_contradiction = test_case.get("has_contradiction", False)
        
        print(f"\n{'-'*70}")
        print(f"[{system_name}] Test {idx}/{len(test_cases)}")
        print(f"Draft: {draft[:80]}...")
        print(f"Expected: {expected_label}")
        print(f"{'-'*70}")
        
        # Run the system
        state0: BullshitState = {
            "user_id": user_id,
            "draft": draft,
            "user_contexts": [],
            "external_contexts": [],
            "combined_contexts": [],
            "nli_contradictions": [],
            "nli_scores": [],
            "llm_response": "",
            "tool_calls_made": []
        }
        
        try:
            result = graph.invoke(state0)
            
            # Extract predictions
            llm_resp = result.get("llm_response", {})
            if isinstance(llm_resp, str):
                llm_resp = {
                    "classification": "unknown",
                    "tweet_summary": llm_resp[:100],
                    "context_summary": "",
                    "confidence": "low",
                    "internal_contradictions": [],
                    "external_contradictions": [],
                    "reasoning": "",
                    "suggested_revision": ""
                }
            
            predicted_label = llm_resp.get("classification", "unknown")
            context_count = len(result.get("combined_contexts", []))
            nli_contradictions = len(result.get("nli_contradictions", []))
            tool_calls = result.get("tool_calls_made", [])
            
            # Check correctness
            is_correct = (predicted_label == expected_label)
            metrics["correct"] += int(is_correct)
            metrics["total"] += 1
            metrics["avg_tool_calls"] += len(tool_calls)
            
            # Track by label
            if expected_label in metrics["by_label"]:
                metrics["by_label"][expected_label]["total"] += 1
                if is_correct:
                    metrics["by_label"][expected_label]["correct"] += 1
            
            # Context usage
            if context_count > 0:
                metrics["context_used"] += 1
            
            # Contradiction detection
            if has_contradiction:
                metrics["contradiction_expected"] += 1
                if nli_contradictions > 0:
                    metrics["contradiction_detected"] += 1
            
            # Display detailed results
            print(f"\n✓ Result: {predicted_label} (expected: {expected_label}) - {'✅ CORRECT' if is_correct else '❌ WRONG'}")
            print(f"  Confidence: {llm_resp.get('confidence', 'unknown')}")
            print(f"  Tweet Summary: {llm_resp.get('tweet_summary', 'N/A')[:1000]}")
            print(f"  Context Summary: {llm_resp.get('context_summary', 'N/A')[:1000]}")
            
            if llm_resp.get('internal_contradictions'):
                print(f"  Internal Contradictions: {len(llm_resp['internal_contradictions'])}")
                for ic in llm_resp['internal_contradictions'][:2]:
                    print(f"    - {ic[:80]}...")
            
            if llm_resp.get('external_contradictions'):
                print(f"  External Contradictions: {len(llm_resp['external_contradictions'])}")
                for ec in llm_resp['external_contradictions'][:2]:
                    if isinstance(ec, dict):
                        print(f"    - {ec.get('claim', '')[:60]} | Source: {ec.get('source', 'N/A')[:40]}")
            
            print(f"  Reasoning: {llm_resp.get('reasoning', 'N/A')[:150]}...")
            
            if llm_resp.get('suggested_revision'):
                print(f"  Suggested Revision: {llm_resp['suggested_revision'][:80]}...")
            
            print(f"  Tools Used: {', '.join(tool_calls)}")
            print(f"  Context Items: {context_count}")
            
            results.append({
                "draft": draft,
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": is_correct,
                "context_count": context_count,
                "nli_contradictions": nli_contradictions,
                "tool_calls": tool_calls,
                "date": test_case.get("date", ""),
                "likes": test_case.get("likes", 0),
                "retweets": test_case.get("retweets", 0),
                "llm_response": llm_resp
            })
            
        except Exception as e:
            print(f"❌ Error evaluating: {e}")
            import traceback
            traceback.print_exc()
            metrics["total"] += 1
            results.append({
                "draft": draft,
                "expected": expected_label,
                "error": str(e)
            })
    
    # Calculate metrics
    accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    context_usage = metrics["context_used"] / metrics["total"] if metrics["total"] > 0 else 0
    contradiction_recall = (metrics["contradiction_detected"] / metrics["contradiction_expected"] 
                           if metrics["contradiction_expected"] > 0 else 0)
    avg_tools = metrics["avg_tool_calls"] / metrics["total"] if metrics["total"] > 0 else 0
    
    # Calculate per-label accuracy
    label_accuracy = {}
    for label, stats in metrics["by_label"].items():
        if stats["total"] > 0:
            label_accuracy[label] = stats["correct"] / stats["total"]
        else:
            label_accuracy[label] = 0
    
    return {
        "system": system_name,
        "user_id": user_id,
        "accuracy": accuracy,
        "context_usage_rate": context_usage,
        "contradiction_recall": contradiction_recall,
        "avg_tool_calls": avg_tools,
        "label_accuracy": label_accuracy,
        "metrics": metrics,
        "results": results
    }

def run_full_evaluation(data_dir: str = "data"):
    """Run evaluation on all CSV timelines with all three systems."""
    systems = {
        "Evidence-First (Predefined)": build_evidence_first_graph(),
        "ReACT (Agent-Driven)": build_react_graph(),
        "Inference-First (Baseline)": build_inference_first_graph()
    }
    
    # Load all CSV timelines
    timelines = load_all_timelines(data_dir)
    
    if not timelines:
        print(f"\n❌ No valid timelines found in '{data_dir}'")
        print("Please ensure CSV files are in the data directory with columns: text, LABEL")
        return {}
    
    print(f"\n✅ Loaded {len(timelines)} timelines")
    all_results = {}
    
    for timeline_data in timelines:
        user_id = timeline_data["user_id"]
        
        print(f"\n{'='*70}")
        print(f"EVALUATING USER: {user_id}")
        print(f"{'='*70}")
        
        timeline_results = {}
        
        for system_name, graph in systems.items():
            print(f"\n{'#'*70}")
            print(f"SYSTEM: {system_name}")
            print(f"{'#'*70}")
            result = evaluate_system(system_name, graph, timeline_data)
            timeline_results[system_name] = result
        
        all_results[user_id] = timeline_results
    
    # Print comprehensive summary
    print(f"\n\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for user_id, timeline_results in all_results.items():
        print(f"\n📊 User: {user_id}")
        print("-" * 80)
        print(f"{'System':<35} {'Overall':<10} {'Bullshit':<12} {'Not BS':<12} {'Ambiguous':<12} {'Avg Tools'}")
        print("-" * 80)
        
        for system_name, result in timeline_results.items():
            label_acc = result.get('label_accuracy', {})
            bs_acc = label_acc.get('bullshit', 0)
            not_bs_acc = label_acc.get('not_bullshit', 0)
            amb_acc = label_acc.get('contextually_ambiguous', 0)
            
            print(f"{system_name:<35} {result['accuracy']:>8.1%} "
                  f"{bs_acc:>10.1%} {not_bs_acc:>10.1%} {amb_acc:>10.1%} "
                  f"{result['avg_tool_calls']:>10.1f}")
    
    # Overall statistics
    print(f"\n\n{'='*80}")
    print("OVERALL STATISTICS ACROSS ALL USERS")
    print(f"{'='*80}")
    
    for system_name in systems.keys():
        all_accuracies = [r[system_name]['accuracy'] for r in all_results.values()]
        all_tool_counts = [r[system_name]['avg_tool_calls'] for r in all_results.values()]
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
        avg_tools = np.mean(all_tool_counts) if all_tool_counts else 0
        
        print(f"\n{system_name}:")
        print(f"  Average Accuracy: {avg_accuracy:.1%}")
        print(f"  Average Tool Calls: {avg_tools:.1f}")
        
        # Collect label-specific accuracies
        label_accs = {'bullshit': [], 'not_bullshit': [], 'contextually_ambiguous': []}
        for user_results in all_results.values():
            user_label_acc = user_results[system_name].get('label_accuracy', {})
            for label in label_accs.keys():
                if user_label_acc.get(label, 0) > 0:  # Only count if there were samples
                    label_accs[label].append(user_label_acc[label])
        
        for label, accs in label_accs.items():
            if accs:
                print(f"  {label.replace('_', ' ').title()} Accuracy: {np.mean(accs):.1%}")
    
    # Save detailed results
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_results = json.loads(json.dumps(all_results, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        json.dump(json_results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_path}")
    
    # Save summary CSV
    summary_rows = []
    for user_id, timeline_results in all_results.items():
        for system_name, result in timeline_results.items():
            row = {
                'user_id': user_id,
                'system': system_name,
                'accuracy': result['accuracy'],
                'avg_tool_calls': result['avg_tool_calls'],
                'context_usage_rate': result.get('context_usage_rate', 0),
                **{f'{label}_accuracy': result.get('label_accuracy', {}).get(label, 0) 
                   for label in ['bullshit', 'not_bullshit', 'contextually_ambiguous']}
            }
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV saved to {summary_path}")
    
    return all_results

# -----------------------------
# 10. Main Execution
# -----------------------------

if __name__ == "__main__":
    print("="*70)
    print("BULLSHIT DETECTOR: ARCHITECTURAL ABLATION STUDY")
    print("="*70)
    print("\n📋 Three Architectures Being Compared:\n")
    print("1️⃣  EVIDENCE-FIRST (Predefined Pipeline)")
    print("   Workflow: retrieve → NLI → assess")
    print("   - Fixed sequence of steps")
    print("   - Uses LLM for query generation")
    print("   - Always runs NLI contradiction detection")
    print("   - Predictable, comprehensive verification\n")
    
    print("2️⃣  REACT (Agent-Driven)")
    print("   Workflow: Agent decides → fetch_wikipedia | fetch_bbc | run_nli | assess")
    print("   - Dynamic tool selection")
    print("   - Agent chooses when to stop")
    print("   - Can skip unnecessary steps")
    print("   - Adaptive, potentially more efficient\n")
    
    print("3️⃣  INFERENCE-FIRST (Baseline)")
    print("   Workflow: direct LLM judgment only")
    print("   - No tools, no retrieval")
    print("   - Pure model knowledge")
    print("   - Fast but potentially unreliable")
    print("   - Baseline for measuring tool value\n")
    
    print("="*70)
    print("Starting evaluation...")
    print("="*70)
    
    results = run_full_evaluation()