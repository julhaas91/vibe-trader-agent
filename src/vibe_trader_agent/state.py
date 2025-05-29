"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Complete agent state storing user-provided information.
    
    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    # Control flow
    user_input: str = field(default="")                 # human-in-the-loop
    is_last_step: IsLastStep = field(default=False)
    next: Optional[str] = field(default=None)           # next node to route to
    
    # User profile data
    profile_complete: bool = field(default=False)       # flag indicating completeness
    name: str = field(default="")
    age: int = field(default=0)
    start_portfolio: float = field(default=0.0)
    planning_horizon: str = field(default="")
    maximum_drawdown_percentage: float = field(default=0.0)
    worst_day_decline_percentage: float = field(default=0.0)
    cash_reserve: float = field(default=0.0)
    max_single_asset_allocation_percentage: float = field(default=0.0)
    target_amount: float = field(default=0.0)
    
    # Investment mandate data
    mandate_complete: bool = field(default=False)       # flag indicating completeness

    existing_holdings: List[Dict[str, Any]] = field(default_factory=list)
    """User's current asset holdings with ticker, quantity, exchange, and region."""
    
    excluded_assets: List[Dict[str, Any]] = field(default_factory=list)
    """Assets to exclude with ticker/category, reason, exchange, and region."""
    
    investment_preferences: List[Dict[str, Any]] = field(default_factory=list)
    """Investment preferences with preference_type and description."""

    # Investment portfolio
    tickers: List[str] = field(default_factory=list)
    """List of tickers available to build a portfolio for."""

    # Black-Litterman modelling inputs
    bl_views: Dict[str, Any] = field(default_factory=dict)
    """
    Contains:
    - p_matrix: List[List[int]] - Views matrix (v×k): Each row represents one view.
    - q_vector: List[float] - Expected returns (v×1) for each view.
    - sigma_vector: List[List[Union[int, float]]] - Diagonal matrix (v×v)of view uncertainty.
    - explanation: str - Brief explanation of the generated views.
    - tickers: List[str] - ticker ordering for reference
    """
    
    # Optimization Engine results
    optimizer_raw_results: Dict[str, Any] = field(default_factory=dict)
    optimizer_outcome: str = field(default="")

    # URL with the final Dashboard
    pdf_dashboard_url: str = field(default="")