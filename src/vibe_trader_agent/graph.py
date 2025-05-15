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
from vibe_trader_agent.nodes import profile_builder, financial_advisor, route_model_output, world_discovery

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node("profile_builder", profile_builder)
builder.add_node("financial_advisor", financial_advisor)
builder.add_node("world_discovery", world_discovery)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as profile_builder
# This means that this node is the first one called
builder.add_edge(START, "profile_builder")

def route_profile_builder(state: State) -> Literal["financial_advisor", "profile_builder", "__end__"]:
    """handles conditional logic to determine next step for profile builder node."""
    if state.next == "financial_advisor":
        return state.next
    return END

builder.add_conditional_edges(
    "profile_builder",
    route_profile_builder
)

def route_financial_adviser(state: State) -> Literal["world_discovery", "tools", "__end__"]:
    """handles conditional logic to determine next step for financial adviser node."""

    last_message = state.messages[-1]

    if state.next == "world_discovery":
        return state.next
    if not last_message.tool_calls:
        return END
    return "tools"

builder.add_conditional_edges(
    "financial_advisor",
    route_financial_adviser,
)

builder.add_conditional_edges(
    "world_discovery",
    route_model_output
    )


# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "financial_advisor")
builder.add_edge("tools", "world_discovery")

# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")
