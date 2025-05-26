"""Vibe-Trader Multi-Agent Setup."""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.nodes import (
    asset_finder,
    financial_advisor,
    human_input_node,
    profile_builder,
    views_analyst,
    optimizer,
)
from vibe_trader_agent.routers import (
    route_asset_finder_output,
    route_financial_advisor_output,
    route_profile_builder_output,
    route_views_analyst_output,
)
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import (
    asset_finder_tools,
    financial_advisor_tools,
    profile_builder_tools,
    views_analyst_tools,
)

# -----

def dummy_node(state: State) -> Dict[str, Any]:
    """Create dummy node placeholder for quick experiments."""
    return {"messages": ["Dummy Node triggered!"]}
# builder.add_node("asset_finder", dummy_node)


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("profile_builder", profile_builder)
builder.add_node("human_input_builder", human_input_node)
builder.add_node("tools_builder", ToolNode(profile_builder_tools))

builder.add_node("financial_advisor", financial_advisor)
builder.add_node("human_input_advisor", human_input_node)
builder.add_node("tools_advisor", ToolNode(financial_advisor_tools))

builder.add_node("asset_finder", asset_finder)
builder.add_node("human_input_finder", human_input_node)
builder.add_node("tools_finder", ToolNode(asset_finder_tools))

builder.add_node("views_analyst", views_analyst)
builder.add_node("tools_analyst", ToolNode(views_analyst_tools))

builder.add_node("optimizer", optimizer)


# Define the flow
builder.add_edge(START, "profile_builder")

# //
builder.add_conditional_edges(
    "profile_builder",
    route_profile_builder_output,
)
builder.add_edge("human_input_builder", "profile_builder")
builder.add_edge("tools_builder", "profile_builder")

# if explicit edge - failing
# builder.add_edge("profile_builder", "financial_advisor")

# //
builder.add_conditional_edges(
    "financial_advisor",
    route_financial_advisor_output,
)
builder.add_edge("human_input_advisor", "financial_advisor")
builder.add_edge("tools_advisor", "financial_advisor")

# //
builder.add_conditional_edges(
    "asset_finder",
    route_asset_finder_output,
)
builder.add_edge("human_input_finder", "asset_finder")
builder.add_edge("tools_finder", "asset_finder")

# //
builder.add_conditional_edges(
    "views_analyst",
    route_views_analyst_output,
)
builder.add_edge("tools_analyst", "views_analyst")

# //
builder.add_edge("optimizer", END)


# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")

