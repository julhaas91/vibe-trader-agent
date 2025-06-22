"""Routers for each node."""

from typing import Literal

from langchain_core.messages import AIMessage

from vibe_trader_agent.state import State


def route_profiler_output(state: State) -> Literal["mandate_strategist", "profiler_tools", "profiler_human"]:
    """Route based on the profile builder output.
    
    Args:
        state (State): Current conversation state
        
    Returns:
        str: Next node to execute
    """
    # Check if routing is explicitly set
    if state.next:
        return state.next  # type: ignore
    
    # Fallback: check if profile is complete
    if state.profile_complete:
        return "mandate_strategist"
    
    # Default: continue with user input
    return "profiler_human"


def route_mandate_strategist_output(state: State) -> Literal["asset_researcher", "strategist_tools", "strategist_human"]:
    """Route based on the financial advisor output.
    
    Args:
        state (State): Current conversation state
        
    Returns:
        str: Next node to execute
    """    
    # Check if routing is explicitly set
    if state.next:
        return state.next  # type: ignore
    
    # Fallback: check if mandate is complete
    if state.mandate_complete:
        return "asset_researcher"
    
    # Default: continue with user input
    return "strategist_human"


def route_asset_researcher_output(state: State) -> Literal["portfolio_analyst", "researcher_tools", "researcher_human"]:
    """Route based on the asset finder output.
    
    Args:
        state (State): Current conversation state
    
    Returns:
        str: Next node to execute
    """    
    # Check if routing is explicitly set
    if state.next:
        return state.next  # type: ignore
    
    # Fallback: check if mandate is complete
    if state.tickers:
        return "portfolio_analyst"
    
    # Default: continue with user input
    return "researcher_human"


def route_portfolio_analyst_output(state: State) -> Literal["analyst_tools", "portfolio_optimizer"]:
    """Route based on the views analyst output.
    
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
        return "analyst_tools"
    
    # END the Graph
    return "portfolio_optimizer"

