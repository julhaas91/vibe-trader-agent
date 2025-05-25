"""Routers for each node."""

from typing import Optional, Dict, Any, Literal
from langchain_core.messages import AIMessage

from vibe_trader_agent.state import State


def route_profile_builder_output(state: State) -> Literal["financial_advisor", "tools_builder", "human_input_builder"]:
    """
    Route based on the profile builder output.
    
    Args:
        state (State): Current conversation state
        
    Returns:
        str: Next node to execute
    """
    # Check if routing is explicitly set
    if state.next:
        return state.next
    
    # Fallback: check if profile is complete
    if state.profile_complete:
        return "financial_advisor"
    
    # Default: continue with user input
    return "human_input_builder"


def route_financial_advisor_output(state: State) -> Literal["asset_finder", "tools_advisor", "human_input_advisor"]:
    """
    Route based on the financial advisor output.
    
    Args:
        state (State): Current conversation state
        
    Returns:
        str: Next node to execute
    """    
    # Check if routing is explicitly set
    if state.next:
        return state.next
    
    # Fallback: check if mandate is complete
    if state.mandate_complete:
        return "asset_finder"
    
    # Default: continue with user input
    return "human_input_advisor"


def route_asset_finder_output(state: State) -> Literal["views_analyst", "tools_finder", "human_input_finder"]:
    """
    Route based on the asset finder output.
    
    Args:
        state (State): Current conversation state
    
    Returns:
        str: Next node to execute
    """    
    # Check if routing is explicitly set
    if state.next:
        return state.next
    
    # Fallback: check if mandate is complete
    if state.tickers:
        return "views_analyst"
    
    # Default: continue with user input
    return "human_input_finder"


def route_views_analyst_output(state: State) -> Literal["tools_analyst", "optimization"]:
    """
    Route based on the views analyst output.
    
    Args:
        state (State): Current conversation state
    
    Returns:
        str: Next node to execute
    """    
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    # If there is tool call, redirect to tools
    if last_message.tool_calls:
        return "tools_analyst"
    
    # END the Graph
    return "optimization"

