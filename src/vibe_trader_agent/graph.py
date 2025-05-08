"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import TOOLS
from vibe_trader_agent.nodes import profile_builder, financial_advisor
from langgraph.graph import StateGraph

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node(profile_builder, "profile_builder")
builder.add_node(financial_advisor, "financial_advisor")
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as profile_builder
# This means that this node is the first one called
builder.add_edge("__start__", "profile_builder")
# builder.add_edge("profile_builder", "financial_advisor")
builder.add_edge("profile_builder", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="Vibe Trader Agent")
