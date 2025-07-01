import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
import json

load_dotenv()
DB_URI = os.getenv("MONGO_URI")

SYSTEM_PROMPT= """You are a an advanced autonomous research agent. Given a high-level topic (e.g., a company or a technology), you must:
1.	Plan: Dynamically create a research plan.
2.	Execute: Use a set of tools in parallel to execute that plan.
3.	Synthesize: Aggregate the findings into a coherent report.
4.	Self-Correct: Identify low-confidence claims in your own reports, pause to ask a human for guidance, and then perform targeted corrective actions based on the feedback.
"""
#hitl tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {"query": query})  # This saves the state in DB and kills the graph
    return human_response["data"]

#tools for research
@tool()
def research_tool(topic:str):
    """This tool does research of the topick mentioned"""
    return "nvidia"

@tool()
def search_news(topic:str):
    """This tool searches the web to get the news related to the topic mentioned"""
    pass

@tool()
def get_financials(topic:str):
    """This tool searches the web to get the financials related to the topic mentioned"""
    pass

tools=[research_tool,search_news,get_financials]


# Initialize your LLM
llm = ChatOpenAI(model='gpt-4.1-mini')
llm_with_tools = llm.bind_tools(tools)

#state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the node
def research_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Graph builder
graph_builder = StateGraph(State)

graph_builder.add_node('research_node', research_node)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, 'research_node')
graph_builder.add_conditional_edges(
    "research_node",
    tools_condition,
)
graph_builder.add_node('tools', "research_node")

# Function to compile graph with MongoDB checkpoint
def compile_mongo_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

def main():
    prompt = input("Enter the research topic > ")
    config = {"configurable": {"thread_id": "1"}}

    with MongoDBSaver.from_conn_string(DB_URI, db_name="checkpointer") as checkpointer:
        graph_with_mongo = compile_mongo_checkpointer(checkpointer)

        # graph_result = graph_with_mongo.invoke(
        #     {"messages": [{"role": "user", "content": prompt}]},
        #     config
        # )
        state=State(messages=[{"role":"user","content":prompt}])
        for event in graph_with_mongo.stream(state,config=config,stream_mode='values'):
            if "messages" in event:
                event["messages"][-1].pretty_print()

        # print("Connecting to:", DB_URI)
        # print(graph_result)

main()