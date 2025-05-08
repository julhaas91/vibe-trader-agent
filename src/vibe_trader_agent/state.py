"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, List

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
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """
    
    # Store extracted profile information
    name: str = field(default="")
    """The user's full name"""
    
    age: int = field(default=0)
    """The user's age in years"""
    
    start_portfolio: float = field(default=0.0)
    """The initial capital the user has ready to invest"""
    
    planning_horizon: str = field(default="")
    """The time period (in months or years) for which the user is planning to invest"""
    
    maximum_drawdown_percentage: float = field(default=0.0)
    """The maximum portfolio decline (in %) the user is comfortable with"""
    
    worst_day_decline_percentage: float = field(default=0.0)
    """The maximum single-day decline (in %) the user can tolerate"""
    
    cash_reserve: float = field(default=0.0)
    """The amount of money the user wants to keep available for immediate withdrawal"""
    
    max_single_asset_allocation_percentage: float = field(default=0.0)
    """The maximum percentage of their portfolio they want in any single asset"""
    
    target_amount: float = field(default=0.0)
    """The financial goal or target amount the user aims to achieve"""
    
    # Store extracted investment information
    existing_holdings: List[Dict[str, Any]] = field(default_factory=list)
    """
    List of dictionaries containing information about assets the user already holds.
    Each dictionary contains:
    - ticker_name: str - The ticker symbol of the asset
    - quantity: float - The quantity of the asset held
    - exchange: str (optional) - The exchange where the asset is traded
    - region: str (optional) - The region where the asset is traded
    """

    excluded_assets: List[Dict[str, Any]] = field(default_factory=list)
    """
    List of dictionaries containing information about assets the user wants to exclude.
    Each dictionary contains:
    - ticker_name: str - The ticker symbol or category to exclude
    - reason: str - The user's reason for exclusion
    - exchange: str (optional) - The exchange where the asset is traded
    - region: str (optional) - The region where the asset is traded
    """

    investment_preferences: List[Dict[str, Any]] = field(default_factory=list)
    """
    List of dictionaries containing the user's investment preferences.
    Each dictionary contains:
    - preference_type: str - The type of preference (e.g., sector, theme, characteristic)
    - description: str - Detailed description of the preference
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
