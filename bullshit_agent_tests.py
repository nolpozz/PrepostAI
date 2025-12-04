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
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "")
    except Exception as e:
        print(f"Wikipedia fetch error: {e}")
    return ""

def fetch_bbc(query: str, max_results: int = 2) -> List[str]:
    """Fetch BBC news articles for a query."""
    url = f"https://www.bbc.co.uk/search?q={query.replace(' ', '+')}"
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for a in soup.select("article h1 a")[:max_results]:
            href = a.get("href", "")
            if not href.startswith("http"):
                href = "https://www.bbc.co.uk" + href
            headline = a.get_text(strip=True)
            try:
                art = requests.get(href, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                art_soup = BeautifulSoup(art.text, "html.parser")
                paras = " ".join(p.get_text() for p in art_soup.find_all("p")[:3])
                results.append(f"{headline}: {paras}")
            except Exception:
                continue
        return results
    except Exception as e:
        print(f"BBC fetch error: {e}")
        return []

# -----------------------------
# 4. NLI SCORING
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

# -----------------------------
# 6. ABLATION 1: EVIDENCE-FIRST (Predefined Pipeline)
# -----------------------------
# DESIGN: Fixed sequence of steps - always retrieve user history, 
# fetch external context, run NLI, then assess. No agent decision-making.

def evidence_first_retrieve_node(state: BullshitState) -> BullshitState:
    """
    PREDEFINED STEP 1: Retrieve user history and external context.
    No LLM decision-making - just execute the retrieval.
    """
    user_id = state["user_id"]
    draft = state["draft"]
    
    print("[Evidence-First] Step 1: Retrieving user history...")
    # Retrieve from user history (predefined action)
    user_contexts = retrieve_similar(user_id, draft, top_k=5)
    state["tool_calls_made"].append("retrieve_user_history")
    
    print("[Evidence-First] Step 2: Fetching external context...")
    # Simple keyword extraction - no LLM planning
    terms = draft.split()[:5]
    query = " ".join(terms)
    
    external_contexts = []
    wiki = fetch_wikipedia(query)
    if wiki:
        external_contexts.append(f"Wikipedia: {wiki}")
        state["tool_calls_made"].append(f"fetch_wikipedia({query})")
    
    bbc = fetch_bbc(query, max_results=2)
    external_contexts.extend(bbc)
    if bbc:
        state["tool_calls_made"].append(f"fetch_bbc({query})")
    
    state["user_contexts"] = user_contexts
    state["external_contexts"] = external_contexts
    state["combined_contexts"] = user_contexts + external_contexts
    return state

def nli_node(state: BullshitState) -> BullshitState:
    """
    PREDEFINED STEP 2: Run NLI to detect contradictions.
    Always executed in Evidence-First pipeline.
    """
    print("[Evidence-First] Step 3: Running NLI contradiction detection...")
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
    PREDEFINED STEP 3: LLM assessment with all retrieved context.
    Always the final step in Evidence-First pipeline.
    """
    print("[Evidence-First] Step 4: Final LLM assessment...")
    ctx_for_prompt = "\n".join(f"- {c}" for c in state["combined_contexts"][:10])
    
    prompt = f"""You are a thoughtful assistant analyzing a tweet draft.

Draft Tweet:
\"\"\"{state['draft']}\"\"\"

Context:
{ctx_for_prompt}

NLI Contradictions Found: {len(state.get('nli_contradictions', []))}

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
        state["llm_response"] = {"summary": raw, "risk": "unknown", "reasoning": "Parse failed", "suggested_revision": ""}
    else:
        state["llm_response"] = parsed
    
    state["tool_calls_made"].append("llm_assess")
    return state

# Build Evidence-First Graph (Predefined Pipeline)
def build_evidence_first_graph():
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

# -----------------------------
# 7. ABLATION 2: REACT (Agent-Driven Tool Selection)
# -----------------------------
# DESIGN: Agent decides which tools to use and when to stop.
# Can choose to: fetch Wikipedia, fetch BBC, run NLI, or finish.

def react_agent_node(state: BullshitState) -> BullshitState:
    """
    REACT: Agent decides which verification steps to take.
    LLM chooses tools dynamically based on what it needs.
    """
    user_id = state["user_id"]
    draft = state["draft"]
    accumulated_context: List[str] = []
    tool_log = state.get("tool_calls_made", [])
    
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
{accumulated_context}

Tools used so far: {tool_log}

Available tools:
- fetch_wikipedia: Get Wikipedia summary for a topic
- fetch_bbc: Get recent news articles
- run_nli: Check for contradictions with existing context
- assess: Make final judgment and finish

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
        
        # Execute the chosen tool
        if action == "fetch_wikipedia":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            wiki = fetch_wikipedia(query)
            if wiki:
                accumulated_context.append(f"Wikipedia({query}): {wiki}")
                tool_log.append(f"fetch_wikipedia({query})")
        
        elif action == "fetch_bbc":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            bbc_items = fetch_bbc(query, max_results=2)
            for item in bbc_items:
                accumulated_context.append(f"BBC({query}): {item}")
            if bbc_items:
                tool_log.append(f"fetch_bbc({query})")
        
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
    
    # Store final state
    state["user_contexts"] = user_contexts
    state["external_contexts"] = accumulated_context[len(user_contexts):]
    state["combined_contexts"] = accumulated_context
    state["tool_calls_made"] = tool_log
    
    return state

def react_assess_node(state: BullshitState) -> BullshitState:
    """Final assessment after ReACT tool selection."""
    print("[ReACT] Making final assessment...")
    ctx_for_prompt = "\n".join(f"- {c}" for c in state["combined_contexts"][:10])
    
    prompt = f"""You are a thoughtful assistant analyzing a tweet draft.

Draft Tweet:
\"\"\"{state['draft']}\"\"\"

Context gathered through agent exploration:
{ctx_for_prompt}

Tools used: {state.get('tool_calls_made', [])}

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
        state["llm_response"] = {"summary": raw, "risk": "unknown", "reasoning": "Parse failed", "suggested_revision": ""}
    else:
        state["llm_response"] = parsed
    
    state["tool_calls_made"].append("llm_assess")
    return state

def build_react_graph():
    graph = StateGraph(BullshitState)
    graph.add_node("react", react_agent_node)
    graph.add_node("assess", react_assess_node)
    
    # Agent-driven flow: react node makes decisions, then final assessment
    graph.add_edge(START, "react")
    graph.add_edge("react", "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# -----------------------------
# 8. ABLATION 3: INFERENCE-FIRST (Non-Agentic Baseline)
# -----------------------------
# DESIGN: No tools, no retrieval. Pure LLM judgment.

def inference_first_node(state: BullshitState) -> BullshitState:
    """Direct LLM judgment with no retrieval or tools."""
    draft = state["draft"]
    
    print("[Inference-First] Making direct judgment with no tools...")
    
    prompt = f"""You are analyzing a tweet draft. Make a judgment based solely on your internal knowledge.

Draft Tweet:
\"\"\"{draft}\"\"\"

Without retrieving any external information or using any tools, assess:
- Is this claim clear and well-grounded?
- Does it seem contradictory or overconfident?
- What risks does it pose?

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
        state["llm_response"] = {"summary": raw, "risk": "unknown", "reasoning": "Parse failed", "suggested_revision": ""}
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
    graph = StateGraph(BullshitState)
    graph.add_node("assess", inference_first_node)
    
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# -----------------------------
# 9. EVALUATION FRAMEWORK
# -----------------------------

def load_timeline(filepath: str) -> Dict[str, Any]:
    """Load a timeline JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_system(system_name: str, graph, timeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a system on a timeline."""
    user_id = timeline_data.get("user_id", "test_user")
    history = timeline_data.get("history", [])
    test_cases = timeline_data.get("test_cases", [])
    
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
        "avg_tool_calls": 0
    }
    
    for test_case in test_cases:
        draft = test_case["draft"]
        expected_label = test_case["label"]
        has_contradiction = test_case.get("has_contradiction", False)
        
        print(f"\n{'='*60}")
        print(f"[{system_name}] Evaluating: {draft[:50]}...")
        print(f"{'='*60}")
        
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
                llm_resp = {"risk": "unknown"}
            
            predicted_risk = llm_resp.get("risk", "unknown")
            context_count = len(result.get("combined_contexts", []))
            nli_contradictions = len(result.get("nli_contradictions", []))
            tool_calls = result.get("tool_calls_made", [])
            
            # Check correctness (simplified: match risk level)
            risk_map = {"well-grounded": "low", "under-specified": "medium", 
                       "contradictory": "high", "contextually-ambiguous": "medium"}
            expected_risk = risk_map.get(expected_label, "medium")
            
            is_correct = (predicted_risk.lower() == expected_risk)
            metrics["correct"] += int(is_correct)
            metrics["total"] += 1
            metrics["avg_tool_calls"] += len(tool_calls)
            
            # Context usage
            if context_count > 0:
                metrics["context_used"] += 1
            
            # Contradiction detection
            if has_contradiction:
                metrics["contradiction_expected"] += 1
                if nli_contradictions > 0:
                    metrics["contradiction_detected"] += 1
            
            print(f"Result: {predicted_risk} (expected: {expected_risk}) - {'✓' if is_correct else '✗'}")
            print(f"Tools used: {tool_calls}")
            
            results.append({
                "draft": draft,
                "expected": expected_label,
                "predicted_risk": predicted_risk,
                "correct": is_correct,
                "context_count": context_count,
                "nli_contradictions": nli_contradictions,
                "tool_calls": tool_calls,
                "llm_response": llm_resp
            })
            
        except Exception as e:
            print(f"Error evaluating: {e}")
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
    
    return {
        "system": system_name,
        "accuracy": accuracy,
        "context_usage_rate": context_usage,
        "contradiction_recall": contradiction_recall,
        "avg_tool_calls": avg_tools,
        "metrics": metrics,
        "results": results
    }

def run_full_evaluation(data_dir: str = "data"):
    """Run evaluation on all timelines with all three systems."""
    systems = {
        "Evidence-First (Predefined)": build_evidence_first_graph(),
        "ReACT (Agent-Driven)": build_react_graph(),
        "Inference-First (Baseline)": build_inference_first_graph()
    }
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory '{data_dir}' not found. Creating example...")
        data_path.mkdir(exist_ok=True)
        # Create example timeline
        example = {
            "user_id": "user1",
            "history": [
                "AI helps artists make new kinds of art.",
                "I believe automation should help humans, not replace them.",
                "Technology should augment human creativity, not replace it."
            ],
            "test_cases": [
                {
                    "draft": "AI will replace all artists soon.",
                    "label": "contradictory",
                    "has_contradiction": True
                },
                {
                    "draft": "Generative AI tools can assist in creative workflows.",
                    "label": "well-grounded",
                    "has_contradiction": False
                },
                {
                    "draft": "The robots are taking over everything!",
                    "label": "under-specified",
                    "has_contradiction": False
                }
            ]
        }
        with open(data_path / "timeline1.json", 'w') as f:
            json.dump(example, f, indent=2)
    
    timeline_files = list(data_path.glob("*.json"))
    all_results = {}
    
    for timeline_file in timeline_files:
        print(f"\n{'='*60}")
        print(f"EVALUATING TIMELINE: {timeline_file.name}")
        print(f"{'='*60}")
        
        timeline_data = load_timeline(timeline_file)
        timeline_results = {}
        
        for system_name, graph in systems.items():
            print(f"\n{'#'*60}")
            print(f"RUNNING: {system_name}")
            print(f"{'#'*60}")
            result = evaluate_system(system_name, graph, timeline_data)
            timeline_results[system_name] = result
        
        all_results[timeline_file.stem] = timeline_results
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    for timeline_name, timeline_results in all_results.items():
        print(f"\n📊 Timeline: {timeline_name}")
        print("-" * 70)
        print(f"{'System':<35} {'Accuracy':<12} {'Context%':<12} {'Contra-Recall':<15} {'Avg Tools'}")
        print("-" * 70)
        for system_name, result in timeline_results.items():
            print(f"{system_name:<35} {result['accuracy']:>10.1%} {result['context_usage_rate']:>10.1%} "
                  f"{result['contradiction_recall']:>13.1%} {result['avg_tool_calls']:>10.1f}")
    
    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    return all_results

# -----------------------------
# 10. Main Execution
# -----------------------------

if __name__ == "__main__":
    print("Running full evaluation across all three ablations...")
    print("\n🔬 ARCHITECTURAL COMPARISON:")
    print("1. Evidence-First: Fixed pipeline (retrieve → NLI → assess)")
    print("2. ReACT: Agent chooses tools dynamically")
    print("3. Inference-First: No tools, direct judgment")
    print()
    results = run_full_evaluation()