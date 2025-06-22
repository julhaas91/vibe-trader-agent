"""Vibe-Trader Multi-Agent Setup."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.nodes import (
    asset_researcher,
    mandate_strategist,
    human_input_node,
    portfolio_optimizer,
    profiler,
    portfolio_analyst,
    reporter,
)
from vibe_trader_agent.routers import (
    route_asset_researcher_output,
    route_mandate_strategist_output,
    route_profiler_output,
    route_portfolio_analyst_output,
)
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import (
    researcher_tools,
    strategist_tools,
    profiler_tools,
    analyst_tools,
)

# -----

# def dummy_node(state: State) -> Dict[str, Any]:
#     """Create dummy node placeholder for quick experiments."""
#     return {"messages": ["Dummy Node triggered!"]}
# builder.add_node("asset_researcher", dummy_node)


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("profiler", profiler)
builder.add_node("profiler_human", human_input_node)
builder.add_node("profiler_tools", ToolNode(profiler_tools))

builder.add_node("mandate_strategist", mandate_strategist)
builder.add_node("strategist_human", human_input_node)
builder.add_node("strategist_tools", ToolNode(strategist_tools))

builder.add_node("asset_researcher", asset_researcher)
builder.add_node("researcher_human", human_input_node)
builder.add_node("researcher_tools", ToolNode(researcher_tools))

builder.add_node("portfolio_analyst", portfolio_analyst)
builder.add_node("analyst_tools", ToolNode(analyst_tools))

builder.add_node("portfolio_optimizer", portfolio_optimizer)

builder.add_node("reporter", reporter)


# Define the flow
builder.add_edge(START, "profiler")

# //
builder.add_conditional_edges(
    "profiler",
    route_profiler_output,
)
builder.add_edge("profiler_human", "profiler")
builder.add_edge("profiler_tools", "profiler")

# if explicit edge - failing
# builder.add_edge("profiler", "mandate_strategist")

# //
builder.add_conditional_edges(
    "mandate_strategist",
    route_mandate_strategist_output,
)
builder.add_edge("strategist_human", "mandate_strategist")
builder.add_edge("strategist_tools", "mandate_strategist")

# //
builder.add_conditional_edges(
    "asset_researcher",
    route_asset_researcher_output,
)
builder.add_edge("researcher_human", "asset_researcher")
builder.add_edge("researcher_tools", "asset_researcher")

# //
builder.add_conditional_edges(
    "portfolio_analyst",
    route_portfolio_analyst_output,
)
builder.add_edge("analyst_tools", "portfolio_analyst")

# //
builder.add_edge("portfolio_optimizer", "reporter")

# //
builder.add_edge("reporter", END)


# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")

