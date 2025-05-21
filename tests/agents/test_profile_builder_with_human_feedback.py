
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from vibe_trader_agent.configuration import Configuration  # type: ignore
from vibe_trader_agent.nodes import (  # type: ignore
    profile_builder,
)
from vibe_trader_agent.state import InputState, State  # type: ignore
from vibe_trader_agent.tools import TOOLS  # type: ignore


def human_input_node(state: State):
    """Node that pauses execution to get input from the human user."""

    # Last msg from the model
    model_response = state.messages[-1].content

    # Use interrupt to pause graph execution and wait for user input
    user_response = interrupt(model_response)
    
    # Return the user input to update the state
    return {"user_input": user_response}


def finance_advisor(state: State):
    return {"messages": ["Financial Advisor triggered!"]}


def route_model_output(state: State):
    """Route based on the output of the profile builder."""
    
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # Move to the next agent?
    if state.next == "finance_advisor":
        return "finance_advisor"
    
    # If there is tool call, redirect to tools
    if last_message.tool_calls:
        return "tools"
    
    # Ask user for the input
    return "human_input"

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("profile_builder", profile_builder)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("human_input", human_input_node)
builder.add_node("finance_advisor", finance_advisor)

builder.add_edge(START, "profile_builder")
builder.add_edge("tools", "profile_builder")
builder.add_edge("human_input", "profile_builder")
builder.add_edge("finance_advisor", END)


builder.add_conditional_edges(
    "profile_builder",
    route_model_output,
)


graph = builder.compile(name="Test Profile Builder with Human in the Loop")
