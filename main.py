import operator
import json
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    reserach_topic: str
    plan: Optional[List[str]] = None
    research_results: Annotated[List[Any], operator.add]
    tasks_to_rerun: Optional[List[str]] = None
    draft_report: Optional[str] = None
    low_confidence_claims: Optional[List[Dict]] = None
    human_feedback: Optional[Dict] = None
    final_report: Optional[str] = None

def mock_llm_planner(topic: str) -> List[str]:
    print(f"Generating plan for '{topic}'...")
    if "NVIDIA" in topic:
        return ["search_news:'NVIDIA new GPUs'", "get_financials:'NVDA'"]
    if "Acme Corporation" in topic:
        return ["search_news:'The Acme Corporation products'", "get_financials:'ACME'"]
    if "cold fusion" in topic:
        return ["search_news:'breakthroughs in cold fusion'", "search_news:'cold fusion commercial viability'"]
    return [f"search_news:'latest news on {topic}'"]

def mock_news_search_tool(query: str) -> str:
    print(f"Searching news for '{query}'...")
    if "peer-reviewed" in query:
        return "Source: nature.com - A 2024 study in Nature confirms no evidence of commercially viable cold fusion, citing significant theoretical hurdles."
    if "cold fusion" in query:
        return "Source: unverified_blog.com - A post suggests a commercial breakthrough in cold fusion is expected by 2030, based on anecdotal evidence."
    if "NVIDIA" in query:
        return "Source: Reuters - NVIDIA announced its new Blackwell GPU architecture, promising significant performance gains for AI workloads."
    return "Source: Associated Press - The Acme Corporation, a fictional entity, primarily features in Looney Tunes cartoons."

def mock_financials_tool(ticker: str) -> Dict[str, Any]:
    print(f"Fetching financial for '{ticker}'...")
    if ticker == "NVDA":
        return {"price": 99.0, "P/E": 88, "MarketCap": "100T"}
    if ticker == "ACME":
        raise ValueError("Invalid - ACME is a private or fictional company.")
    return {}

def mock_llm_synthesizer(topic: str, results: List[Any]) -> Dict[str, Any]:
    print("Synthesizing...")
    report_text = f"Draft Report: {topic}\n\n"
    low_confidence_claims = []
    for res in results:
        if isinstance(res, str):
            report_text += f"- {res}\n"
            if "unverified_blog.com" in res:
                claim = {"claim_text": "A commercial breakthrough is expected by 2030.", "source": "unverified_blog.com"}
                low_confidence_claims.append(claim)
        elif isinstance(res, dict):
            report_text += f"- Financials: Price=${res.get('price')}, P/E={res.get('P/E')}, MarketCap={res.get('MarketCap')}\n"
        elif isinstance(res, Exception):
            report_text += f"- An error occurred: {str(res)}\n"
    return {"draft_report": report_text, "low_confidence_claims": low_confidence_claims}

def researchplan_node(state: GraphState):
    print("---(Node: Planner)---")
    topic = state['reserach_topic']
    plan = mock_llm_planner(topic)
    return {"plan": plan, "research_results": []}

def researcher_node_base(state: GraphState, tool_name: str, tool_callable, query_prefix: str):
    tasks = state.get('tasks_to_rerun') or state.get('plan')
    my_tasks = [t for t in tasks if t.startswith(query_prefix)]
    if not my_tasks: return {}
    print(f"---(Node: {tool_name})---")
    results = []
    for task in my_tasks:
        query = task.split(':', 1)[1].strip("'\"")
        try:
            result = tool_callable(query)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in {tool_name}: {e}")
            results.append(e)
    return {"research_results": results}

def newsresearcher_node(state: GraphState):
    return researcher_node_base(state, "News Researcher", mock_news_search_tool, "search_news:")

def financialresearcher_node(state: GraphState):
    return researcher_node_base(state, "Financial Researcher", mock_financials_tool, "get_financials:")


def synthesizer_node(state: GraphState):
    print("-Synthesizer-")
    synthesis = mock_llm_synthesizer(state['reserach_topic'], state['research_results'])
    return {
        "draft_report": synthesis['draft_report'],
        "low_confidence_claims": synthesis['low_confidence_claims'],
        "tasks_to_rerun": None
    }

def human_review_node(state: GraphState):
    print("-Human Review Paused-")
    return {}

def finish_node(state: GraphState):
    print("FINISH")
    return {"final_report": state['draft_report']}

def execute_research_plan_edge(state: GraphState) -> List[str]:
    print("EXECUTE")
    plan = state['plan']
    researcher_nodes = []
    if any("search_news" in task for task in plan): researcher_nodes.append("news_researcher")
    if any("get_financials" in task for task in plan): researcher_nodes.append("financial_researcher")
    print(f"parallel researcher: {researcher_nodes}")
    return researcher_nodes

def should_ask_human_edge(state: GraphState) -> str:
    print("Ask Human")
    if state.get('low_confidence_claims'):
        print("low confidence claim")
        return "ask_human"
    else:
        print("Report finish")
        return "end"

def route_correction_edge(state: GraphState) -> str:
    print("Route Correction")
    feedback = state.get("human_feedback")
    if not feedback:
        print("  No feedback provided. Finishing.")
        return "end"

    human_instruction = feedback.get("human_instruction", "")
    if "news search" in human_instruction:
        tasks_to_rerun = [human_instruction.replace("Re-run ", "")]
        print(f"Feedback received. Re-routing to News Researcher with new tasks: {tasks_to_rerun}")
        return {
            "tasks_to_rerun": tasks_to_rerun,
            "human_feedback": None,
            "research_results": []
        }
    
    print("Feedback didn't specify a re-run task. Finishing.")
    return "end"

memory = MemorySaver()
builder = StateGraph(GraphState)

builder.add_node("planner", researchplan_node)
builder.add_node("news_researcher", newsresearcher_node)
builder.add_node("financial_researcher", financialresearcher_node)
builder.add_node("synthesizer", synthesizer_node)

def process_human_feedback_node(state: GraphState) -> dict:
    return route_correction_edge(state)
builder.add_node("ask_human", process_human_feedback_node)
builder.add_node("end", finish_node)

builder.set_entry_point("planner")
builder.add_conditional_edges("planner", execute_research_plan_edge)
builder.add_edge("news_researcher", "synthesizer")
builder.add_edge("financial_researcher", "synthesizer")
builder.add_conditional_edges("synthesizer", should_ask_human_edge, {
    "ask_human": "ask_human",
    "end": "end"
})

builder.add_conditional_edges("ask_human", lambda x: "news_researcher" if x.get("tasks_to_rerun") else "finish", {
    "news_researcher": "news_researcher",
    "end": "end"
})
builder.add_edge("end", END)

research_agent_app = builder.compile(checkpointer=memory, interrupt_before=["ask_human"])


human_correction = {
        "correction_for_claim": "The claim about a 2030 breakthrough is unverified.",
        "human_instruction": "Re-run news search: 'peer-reviewed papers on cold fusion commercial viability'"
    }

def run_scenario(reserach_topic, human_feedback=None):
    print("\n" + "="*50)
    print(f" TOPIC: {reserach_topic}")
    print("="*50 + "\n")
    config = {"configurable": {"thread_id": "1"}}
    
    # Use stream to run the graph
    for _ in research_agent_app.stream({"reserach_topic": reserach_topic}, config):
        pass

    if research_agent_app.get_state(config).next == ("ask_human",):
        current_state_vals = research_agent_app.get_state(config).values
        print("\n human input nneeded")
        print("Draft Report:")
        print(current_state_vals['draft_report'])
        print("\nLow Confidence Claims:")
        for claim in current_state_vals['low_confidence_claims']:
            print(f"- Claim: '{claim['claim_text']}' (Source: {claim['source']})")

        if human_feedback:
            print("\n HUMAN IN LOOP---")
            print(f"Feedback provided: {human_feedback}")
            for _ in research_agent_app.stream({"human_feedback": human_feedback}, config):
                pass
        else:
            print("\nNO HUMAN FEEDBACK PROVIDED")
            for _ in research_agent_app.stream(None, config):
                pass

    print("\n" + "-"*50)
    print("SCENARIO COMPLETE")
    print("-"*50)

    final_state_values = research_agent_app.get_state(config).values
    print("Final Report:")
    print(final_state_values.get('final_report', "No final report generated."))
    print("\nFinal State:")
    print(json.dumps(final_state_values, indent=2, default=str))


if __name__ == "__main__":
    run_scenario("NVIDIA")
    #run_scenario("The Acme Corporation")
    #run_scenario("The future of cold fusion", human_feedback=human_correction)