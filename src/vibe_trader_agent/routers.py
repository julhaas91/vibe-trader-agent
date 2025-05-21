"""Routers for each node."""

from langchain_core.messages import AIMessage

from vibe_trader_agent.state import State


def route_profile_builder_output(state: State) -> str:
    """Route based on the output of the profile builder."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # Move to the next agent?
    if state.next == "financial_advisor":
        return "financial_advisor"
        
    # Ask user for the input
    return "human_input_profile"


def route_financial_advisor_output(state: State) -> str:
    """Route based on the output of the financial advisor."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # Move to the next agent?
    if state.next == "world_discovery":
        return "world_discovery"
    
    # If there is tool call, redirect to tools
    if last_message.tool_calls:
        return "financial_advisor_tools"
    
    # Ask user for the input
    return "human_input_advisor"


def route_world_discovery_output(state: State) -> str:
    """Route based on the output of the world discovery."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # If there is tool call, redirect to tools
    if last_message.tool_calls:
        return "world_discovery_tools"
    
    # Move to the next agent
    return "views_analyst"


def route_views_analyst_output(state: State) -> str:
    """Route based on the output of the world discovery."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # If there is tool call, redirect to tools
    if last_message.tool_calls:
        return "views_analyst_tools"
    
    # END the Graph
    return "__end__"

