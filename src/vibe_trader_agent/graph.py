"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import TOOLS
from vibe_trader_agent.nodes import profile_builder, financial_advisor, route_model_output

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node(profile_builder, "profile_builder")
builder.add_node(financial_advisor, "financial_advisor")
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as profile_builder
# This means that this node is the first one called
builder.add_edge(START, "profile_builder")

# Add conditional edge from profile_builder to financial_advisor
def route_profile_builder(state: State) -> Literal["financial_advisor", "__end__"]:
    """Route based on profile builder output."""
    if state.next == "financial_advisor":
        return "financial_advisor"
    return "__end__"

builder.add_conditional_edges(
    "profile_builder",
    route_profile_builder,
)

# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "financial_advisor",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "financial_advisor")

# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")