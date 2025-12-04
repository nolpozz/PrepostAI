# ==========================================
# test_search_tools.py
# Test search tool functionality with ReACT and Evidence-First workflows
# ==========================================

from typing import TypedDict, Annotated, List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import json
import re

from openai import OpenAI
from inference_auth_token import get_access_token

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# -----------------------------
# 1. MODEL INITIALIZATION
# -----------------------------

access_token = get_access_token()
chat = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_key=access_token,
    openai_api_base="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
)

# -----------------------------
# 2. SEARCH TOOLS
# -----------------------------

def fetch_wikipedia(query: str) -> str:
    """Fetch Wikipedia summary for a query."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        print(f"  🔍 Fetching Wikipedia: {query}")
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            print(f"  ✅ Wikipedia returned {len(extract)} chars")
            return extract
        else:
            print(f"  ❌ Wikipedia status: {r.status_code}")
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
                        print(f"  ✅ Wikipedia search returned snippet")
                        return snippet
            except Exception as e:
                print(f"  ❌ Wikipedia search API also failed: {e}")
    except Exception as e:
        print(f"  ❌ Wikipedia error: {e}")
    return ""

def fetch_bbc(query: str, max_results: int = 2) -> List[str]:
    """Fetch BBC news articles for a query."""
    url = f"https://www.bbc.co.uk/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        print(f"  🔍 Fetching BBC News: {query}")
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code != 200:
            print(f"  ❌ BBC status: {r.status_code}")
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        
        # Try multiple selectors for BBC's layout
        # Look for article links in search results
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
            
            # For testing, just return headline without fetching full article
            # This avoids making too many requests
            results.append(f"BBC News: {headline}")
            print(f"  ✅ BBC article: {headline[:50]}...")
            
            if len(results) >= max_results:
                break
        
        if not results:
            print(f"  ⚠️  No BBC results found")
        return results
        
    except Exception as e:
        print(f"  ❌ BBC error: {e}")
        return []

def fetch_google_search_mock(query: str) -> List[str]:
    """Mock Google search - in reality you'd use an API."""
    print(f"  🔍 Mock Google Search: {query}")
    return [f"Mock result for '{query}': This would be real search results in production"]

# -----------------------------
# 3. HELPER FUNCTIONS
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
# 4. STATE DEFINITION
# -----------------------------

class SearchTestState(TypedDict):
    draft: str
    context_gathered: Annotated[List[str], "accumulated search results"]
    tool_calls_made: Annotated[List[str], "log of tool calls"]
    assessment: str
    accuracy_report: Dict[str, Any]

# -----------------------------
# 5. EVIDENCE-FIRST (PREDEFINED PIPELINE)
# -----------------------------

def evidence_first_search_node(state: SearchTestState) -> SearchTestState:
    """
    Predefined search pipeline with query generation:
    1. Generate search queries using LLM
    2. Search Wikipedia
    3. Search BBC
    No dynamic decision-making
    """
    print("\n" + "="*60)
    print("EVIDENCE-FIRST: Executing predefined search sequence")
    print("="*60)
    
    draft = state["draft"]
    context = []
    tools = []
    
    # Step 1: Generate search queries
    print("\n[Step 1] Generating optimized search queries...")
    queries = generate_search_queries(draft)
    
    # Step 2: Search Wikipedia for each query
    print("\n[Step 2] Searching Wikipedia...")
    for query in queries[:2]:  # Limit to 2 queries for Wikipedia
        wiki = fetch_wikipedia(query)
        if wiki:
            context.append(f"Wikipedia({query}): {wiki[:500]}...")
            tools.append(f"fetch_wikipedia({query})")
    
    # Step 3: Search BBC for primary query
    print("\n[Step 3] Searching BBC News...")
    if queries:
        bbc = fetch_bbc(queries[0], max_results=2)
        context.extend(bbc)
        if bbc:
            tools.append(f"fetch_bbc({queries[0]})")
    
    state["context_gathered"] = context
    state["tool_calls_made"] = tools
    
    print(f"\n✅ Evidence-First completed: {len(context)} context items gathered")
    return state

def evidence_first_assess_node(state: SearchTestState) -> SearchTestState:
    """Factual accuracy assessment for Evidence-First."""
    print("\n[Assessment] Analyzing factual accuracy...")
    
    accuracy_report = assess_factual_accuracy(state["draft"], state["context_gathered"])
    
    # Create readable assessment
    assessment = f"""Accuracy: {accuracy_report.get('accuracy_rating', 'unknown')} (Confidence: {accuracy_report.get('confidence', 'unknown')})

Key Findings: {accuracy_report.get('key_findings', 'N/A')}

Concerns: {accuracy_report.get('concerns', 'None identified')}"""
    
    state["assessment"] = assessment
    state["accuracy_report"] = accuracy_report
    
    print(f"\n{assessment}")
    return state

def build_evidence_first_graph():
    """Build predefined pipeline graph."""
    graph = StateGraph(SearchTestState)
    graph.add_node("search", evidence_first_search_node)
    graph.add_node("assess", evidence_first_assess_node)
    
    graph.add_edge(START, "search")
    graph.add_edge("search", "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# -----------------------------
# 6. REACT (AGENT-DRIVEN SEARCH)
# -----------------------------

def react_agent_search_node(state: SearchTestState) -> SearchTestState:
    """
    ReACT agent that decides which search tools to use.
    Agent can choose: fetch_wikipedia, fetch_bbc, or finish.
    """
    print("\n" + "="*60)
    print("REACT: Agent-driven search")
    print("="*60)
    
    draft = state["draft"]
    context = []
    tools = []
    failed_attempts = []  # Track failed tool calls to avoid loops
    
    max_rounds = 5
    for round_idx in range(max_rounds):
        print(f"\n{'─'*60}")
        print(f"Round {round_idx + 1}: Agent deciding next action...")
        print(f"{'─'*60}")
        
        # Agent decides what to do
        prompt = f"""You are a research agent gathering information to verify this draft:

Draft: "{draft}"

Context gathered so far:
{context if context else "[None yet]"}

Tools used: {tools if tools else "[None yet]"}
Failed attempts: {failed_attempts if failed_attempts else "[None]"}

Available actions:
- fetch_wikipedia: Get encyclopedia information (use JSON: {{"action": "fetch_wikipedia", "query": "topic"}})
- fetch_bbc: Get recent news (use JSON: {{"action": "fetch_bbc", "query": "topic"}})
- finish: Done gathering info (use JSON: {{"action": "finish"}})

IMPORTANT: If a tool failed, try a different tool or different query. Don't repeat failed attempts.

What should you do next? Respond ONLY with JSON."""
        
        resp = chat.invoke([{"role": "user", "content": prompt}])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = extract_json_from_text(raw)
        
        print(f"Agent response: {raw[:200]}...")
        
        if not parsed or "action" not in parsed:
            print("⚠️  Could not parse agent decision. Finishing.")
            break
        
        action = parsed.get("action")
        print(f"\n🤖 Agent chose action: {action}")
        
        # Check if this exact attempt was already tried and failed
        attempt_key = None
        if action in ["fetch_wikipedia", "fetch_bbc"]:
            query = parsed.get("query", " ".join(draft.split()[:3]))
            attempt_key = f"{action}:{query}"
            if attempt_key in failed_attempts:
                print(f"   ⚠️  This attempt already failed. Agent should try something else.")
                # Give agent one more chance to try something different
                if round_idx < max_rounds - 1:
                    continue
                else:
                    break
        
        # Execute the action
        if action == "fetch_wikipedia":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            print(f"   Query: {query}")
            wiki = fetch_wikipedia(query)
            if wiki:
                context.append(f"Wikipedia({query}): {wiki[:500]}...")
                tools.append(f"fetch_wikipedia({query})")
            else:
                failed_attempts.append(attempt_key)
                print("   ⚠️  No Wikipedia results - agent should try different approach")
        
        elif action == "fetch_bbc":
            query = parsed.get("query", " ".join(draft.split()[:3]))
            print(f"   Query: {query}")
            bbc = fetch_bbc(query, max_results=2)
            if bbc:
                context.extend(bbc)
                tools.append(f"fetch_bbc({query})")
            else:
                failed_attempts.append(attempt_key)
                print("   ⚠️  No BBC results - agent should try different approach")
        
        elif action == "finish":
            print("✅ Agent decided to finish")
            break
        
        else:
            print(f"⚠️  Unknown action: {action}. Finishing.")
            break
        
        # Safety check: if we've had 3+ failed attempts with no success, stop
        if len(failed_attempts) >= 3 and len(context) == 0:
            print("\n⚠️  Multiple failed attempts with no results. Stopping search.")
            break
    
    state["context_gathered"] = context
    state["tool_calls_made"] = tools
    
    print(f"\n✅ ReACT completed: {len(context)} context items gathered using {len(tools)} tool calls")
    return state

def react_assess_node(state: SearchTestState) -> SearchTestState:
    """Factual accuracy assessment for ReACT."""
    print("\n[Assessment] Analyzing factual accuracy...")
    
    accuracy_report = assess_factual_accuracy(state["draft"], state["context_gathered"])
    
    # Create readable assessment
    assessment = f"""Accuracy: {accuracy_report.get('accuracy_rating', 'unknown')} (Confidence: {accuracy_report.get('confidence', 'unknown')})

Key Findings: {accuracy_report.get('key_findings', 'N/A')}

Concerns: {accuracy_report.get('concerns', 'None identified')}

Tools Used: {state.get('tool_calls_made', [])}"""
    
    state["assessment"] = assessment
    state["accuracy_report"] = accuracy_report
    
    print(f"\n{assessment}")
    return state

def build_react_graph():
    """Build ReACT agent graph."""
    graph = StateGraph(SearchTestState)
    graph.add_node("react", react_agent_search_node)
    graph.add_node("assess", react_assess_node)
    
    graph.add_edge(START, "react")
    graph.add_edge("react", "assess")
    graph.add_edge("assess", END)
    
    return graph.compile()

# -----------------------------
# 7. QUERY GENERATION & FACT-CHECKING
# -----------------------------

def generate_search_queries(draft: str) -> List[str]:
    """Use LLM to generate optimal search queries for a draft claim."""
    print(f"\n🤔 Generating search queries for: \"{draft}\"")
    
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
        queries = parsed["queries"]
        print(f"✅ Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        return queries
    else:
        print(f"⚠️  Could not parse queries, using simple fallback")
        # Fallback: extract first 5 words
        return [" ".join(draft.split()[:5])]

def assess_factual_accuracy(draft: str, context: List[str]) -> Dict[str, Any]:
    """Use LLM to assess factual accuracy of draft based on gathered context."""
    print(f"\n🔍 Assessing factual accuracy...")
    
    context_summary = "\n".join(f"- {c[:300]}..." for c in context[:5])
    
    prompt = f"""You are a fact-checker analyzing a claim against gathered evidence.

Claim: "{draft}"

Evidence gathered:
{context_summary}

Assess the claim's factual accuracy:
1. Is it supported by the evidence?
2. Is it contradicted by the evidence?
3. Is there insufficient evidence to determine?
4. Are there important caveats or nuances?

Respond with JSON:
{{
  "accuracy_rating": "supported/contradicted/insufficient/partially_supported",
  "confidence": "high/medium/low",
  "key_findings": "brief summary of what evidence shows",
  "concerns": "any issues, exaggerations, or missing context"
}}

Respond ONLY with JSON."""
    
    resp = chat.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if hasattr(resp, "content") else str(resp)
    parsed = extract_json_from_text(raw)
    
    if not parsed:
        print("⚠️  Could not parse assessment")
        return {
            "accuracy_rating": "unknown",
            "confidence": "low",
            "key_findings": "Failed to parse",
            "concerns": "Assessment failed"
        }
    
    print(f"✅ Accuracy: {parsed.get('accuracy_rating', 'unknown')} "
          f"(Confidence: {parsed.get('confidence', 'unknown')})")
    return parsed

# -----------------------------
# 8. TEST CASES
# -----------------------------

TEST_DRAFTS = [
    {
        "draft": "The 2024 Olympics were held in Paris, France.",
        "description": "Recent factual event - should find Wikipedia and news coverage",
        "expected_accuracy": "supported"
    },
    {
        "draft": "AI language models can generate human-like text.",
        "description": "Technical topic - should find Wikipedia articles",
        "expected_accuracy": "supported"
    },
    {
        "draft": "Climate change is causing more extreme weather events.",
        "description": "News-heavy topic - should find recent BBC articles",
        "expected_accuracy": "supported"
    },
    {
        "draft": "Quantum computers will replace all classical computers by 2025.",
        "description": "Dubious claim - should find contradicting information",
        "expected_accuracy": "contradicted"
    },
    {
        "draft": "The new iPhone has better battery life than previous models.",
        "description": "Product claim - should find tech news",
        "expected_accuracy": "partially_supported"
    },
    {
        "draft": "The Earth is flat and NASA is hiding the truth.",
        "description": "Conspiracy theory - should be strongly contradicted",
        "expected_accuracy": "contradicted"
    },
    {
        "draft": "Scientists discovered a cure for all cancers last week.",
        "description": "False breaking news - should find no supporting evidence",
        "expected_accuracy": "insufficient"
    }
]

# -----------------------------
# 8. TEST RUNNER
# -----------------------------

def test_search_tools():
    """Test individual search tools."""
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL SEARCH TOOLS")
    print("="*70)
    
    # Test Wikipedia
    print("\n--- Testing Wikipedia ---")
    result = fetch_wikipedia("Artificial Intelligence")
    print(f"Result length: {len(result)}")
    print(f"Preview: {result[:200]}...\n")
    
    # Test BBC
    print("\n--- Testing BBC News ---")
    results = fetch_bbc("climate change", max_results=2)
    print(f"Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r[:150]}...\n")

def run_single_test(system_name: str, graph, test_case: Dict[str, str]):
    """Run a single test case."""
    print("\n" + "="*70)
    print(f"TEST CASE: {test_case['description']}")
    print(f"DRAFT: \"{test_case['draft']}\"")
    print(f"EXPECTED ACCURACY: {test_case.get('expected_accuracy', 'unknown')}")
    print("="*70)
    
    state0 = {
        "draft": test_case["draft"],
        "context_gathered": [],
        "tool_calls_made": [],
        "assessment": "",
        "accuracy_report": {}
    }
    
    result = graph.invoke(state0)
    
    print("\n" + "─"*70)
    print(f"RESULTS FOR {system_name}")
    print("─"*70)
    print(f"Context items gathered: {len(result['context_gathered'])}")
    print(f"Tool calls made: {result['tool_calls_made']}")
    
    # Show accuracy assessment
    accuracy_report = result.get('accuracy_report', {})
    expected = test_case.get('expected_accuracy', 'unknown')
    actual = accuracy_report.get('accuracy_rating', 'unknown')
    match = "✅" if expected == actual else "⚠️"
    
    print(f"\n{match} Accuracy Assessment:")
    print(f"   Expected: {expected}")
    print(f"   Actual:   {actual}")
    print(f"   Confidence: {accuracy_report.get('confidence', 'unknown')}")
    
    print(f"\nContext preview:")
    for i, ctx in enumerate(result['context_gathered'][:3], 1):
        print(f"  {i}. {ctx[:200]}...")
    
    print(f"\n{result['assessment']}")
    
    return result

def run_all_tests():
    """Run all test cases with both architectures."""
    print("\n" + "🧪"*35)
    print("SEARCH TOOLS TEST SUITE")
    print("🧪"*35)
    
    # First test individual tools
    test_search_tools()
    
    # Build both graphs
    evidence_first = build_evidence_first_graph()
    react = build_react_graph()
    
    results = {
        "evidence_first": [],
        "react": []
    }
    
    accuracy_stats = {
        "evidence_first": {"correct": 0, "total": 0},
        "react": {"correct": 0, "total": 0}
    }
    
    # Run each test case with both systems
    for test_case in TEST_DRAFTS:
        print("\n\n" + "🔬"*35)
        print(f"COMPARING ARCHITECTURES")
        print("🔬"*35)
        
        expected = test_case.get('expected_accuracy', 'unknown')
        
        # Evidence-First
        print("\n" + "┌" + "─"*68 + "┐")
        print("│  ARCHITECTURE 1: EVIDENCE-FIRST (PREDEFINED PIPELINE)           │")
        print("└" + "─"*68 + "┘")
        ef_result = run_single_test("Evidence-First", evidence_first, test_case)
        results["evidence_first"].append(ef_result)
        
        # Track accuracy
        ef_actual = ef_result.get('accuracy_report', {}).get('accuracy_rating', 'unknown')
        accuracy_stats["evidence_first"]["total"] += 1
        if ef_actual == expected:
            accuracy_stats["evidence_first"]["correct"] += 1
        
        # ReACT
        print("\n" + "┌" + "─"*68 + "┐")
        print("│  ARCHITECTURE 2: REACT (AGENT-DRIVEN)                           │")
        print("└" + "─"*68 + "┘")
        react_result = run_single_test("ReACT", react, test_case)
        results["react"].append(react_result)
        
        # Track accuracy
        react_actual = react_result.get('accuracy_report', {}).get('accuracy_rating', 'unknown')
        accuracy_stats["react"]["total"] += 1
        if react_actual == expected:
            accuracy_stats["react"]["correct"] += 1
        
        # Comparison
        print("\n" + "📊"*35)
        print("COMPARISON")
        print("─"*70)
        print(f"Evidence-First: {len(ef_result['tool_calls_made'])} tool calls, "
              f"{len(ef_result['context_gathered'])} context items")
        print(f"                Accuracy: {ef_actual}")
        print(f"ReACT:          {len(react_result['tool_calls_made'])} tool calls, "
              f"{len(react_result['context_gathered'])} context items")
        print(f"                Accuracy: {react_actual}")
        
        input("\nPress Enter to continue to next test case...")
    
    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    ef_avg_tools = sum(len(r['tool_calls_made']) for r in results['evidence_first']) / len(TEST_DRAFTS)
    ef_avg_context = sum(len(r['context_gathered']) for r in results['evidence_first']) / len(TEST_DRAFTS)
    ef_accuracy = (accuracy_stats["evidence_first"]["correct"] / 
                   accuracy_stats["evidence_first"]["total"] * 100)
    
    react_avg_tools = sum(len(r['tool_calls_made']) for r in results['react']) / len(TEST_DRAFTS)
    react_avg_context = sum(len(r['context_gathered']) for r in results['react']) / len(TEST_DRAFTS)
    react_accuracy = (accuracy_stats["react"]["correct"] / 
                      accuracy_stats["react"]["total"] * 100)
    
    print(f"\nEvidence-First (Predefined):")
    print(f"  Average tool calls: {ef_avg_tools:.1f}")
    print(f"  Average context items: {ef_avg_context:.1f}")
    print(f"  Accuracy match rate: {ef_accuracy:.1f}% ({accuracy_stats['evidence_first']['correct']}/{accuracy_stats['evidence_first']['total']})")
    
    print(f"\nReACT (Agent-Driven):")
    print(f"  Average tool calls: {react_avg_tools:.1f}")
    print(f"  Average context items: {react_avg_context:.1f}")
    print(f"  Accuracy match rate: {react_accuracy:.1f}% ({accuracy_stats['react']['correct']}/{accuracy_stats['react']['total']})")
    
    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)
    
    if react_avg_tools < ef_avg_tools:
        print(f"✓ ReACT is more efficient: {ef_avg_tools - react_avg_tools:.1f} fewer tool calls on average")
    else:
        print(f"✓ Evidence-First is more efficient: {react_avg_tools - ef_avg_tools:.1f} fewer tool calls on average")
    
    if react_accuracy > ef_accuracy:
        print(f"✓ ReACT is more accurate: {react_accuracy - ef_accuracy:.1f}% better")
    elif ef_accuracy > react_accuracy:
        print(f"✓ Evidence-First is more accurate: {ef_accuracy - react_accuracy:.1f}% better")
    else:
        print(f"✓ Both systems equally accurate")
    
    print("\n✅ All tests completed!")
    
    return results

# -----------------------------
# 9. MAIN
# -----------------------------

if __name__ == "__main__":
    print("Starting search tools test...\n")
    results = run_all_tests()