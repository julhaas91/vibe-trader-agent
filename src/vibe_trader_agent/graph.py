"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from typing import Literal

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.nodes import (
    financial_advisor,
    profile_builder,
    route_model_output,
    views_analyst,
    world_discovery,
)
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import TOOLS

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node("profile_builder", profile_builder)
builder.add_node("financial_advisor", financial_advisor)
builder.add_node("world_discovery", world_discovery)
builder.add_node("views_analyst", views_analyst)
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

def route_financial_adviser(state: State) -> Literal["world_discovery", "__end__"]:
    """Route based on financial adviser output."""
    if state.next == "world_discovery":
        return "world_discovery"
    return "__end__"

builder.add_conditional_edges(
    "financial_advisor",
    route_financial_adviser,
)

builder.add_conditional_edges(
    "financial_advisor",
    route_model_output,
)

builder.add_conditional_edges(
    "world_discovery",
    route_model_output
)

def route_world_discovery(state: State) -> Literal["views_analyst", "__end__"]:
    """Route based on financial adviser output."""
    if state.next == "views_analyst":
        return "views_analyst"
    return "__end__"

builder.add_conditional_edges(
    "world_discovery",
    route_world_discovery,
)


builder.add_conditional_edges(
    "views_analyst",
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "financial_advisor")
builder.add_edge("tools", "world_discovery")
builder.add_edge("tools", "views_analyst")

# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")
