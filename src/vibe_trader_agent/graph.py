"""Vibe-trader Multi-Agent setup."""


from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.nodes import (
    financial_advisor,
    human_input_node,
    profile_builder,
    views_analyst,
    world_discovery,
)
from vibe_trader_agent.routers import (
    route_financial_advisor_output,
    route_profile_builder_output,
    route_views_analyst_output,
    route_world_discovery_output,
)
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import (
    financial_advisor_tools,
    views_analyst_tools,
    world_discovery_tools,
)

# -----

# def world_discovery(state: State):
#     print("World Discovery triggered")
#     return {"messages": ["World Discovery triggered!"]}



# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node("profile_builder", profile_builder)
builder.add_node("human_input_profile", human_input_node)

builder.add_node("financial_advisor", financial_advisor)
builder.add_node("human_input_advisor", human_input_node)
builder.add_node("financial_advisor_tools", ToolNode(financial_advisor_tools))

builder.add_node("world_discovery", world_discovery)
builder.add_node("world_discovery_tools", ToolNode(world_discovery_tools))

builder.add_node("views_analyst", views_analyst)
builder.add_node("views_analyst_tools", ToolNode(views_analyst_tools))



# Define workflow
builder.add_edge(START, "profile_builder")

builder.add_conditional_edges(
    "profile_builder",
    route_profile_builder_output,
)
builder.add_edge("human_input_profile", "profile_builder")

# if explicit edge - failing
# builder.add_edge("profile_builder", "financial_advisor")

# //
builder.add_conditional_edges(
    "financial_advisor",
    route_financial_advisor_output,
)
builder.add_edge("human_input_advisor", "financial_advisor")
builder.add_edge("financial_advisor_tools", "financial_advisor")

# //
builder.add_conditional_edges(
    "world_discovery",
    route_world_discovery_output,
)
builder.add_edge("world_discovery_tools", "world_discovery")

# //
builder.add_conditional_edges(
    "views_analyst",
    route_views_analyst_output,
)
builder.add_edge("views_analyst_tools", "views_analyst")


# Compile the builder into an executable graph
graph = builder.compile(name="[Test] Vibe Trader Agent")

