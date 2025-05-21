"""Node module for the Vibe Trader Agent."""

import json
import re
from datetime import UTC, datetime
from typing import Any, Dict, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.misc import extract_json, get_current_date
from vibe_trader_agent.prompts import (
    CONSTRAINTS_EXTRACTOR_SYSTEM_PROMPT,
    VIEWS_ANALYST_SYSTEM_PROMPT,
    WORLD_DISCOVERY_PROMPT,
)
from vibe_trader_agent.state import State
from vibe_trader_agent.tools import (
    financial_advisor_tools,
    views_analyst_tools,
    world_discovery_tools,
)
from vibe_trader_agent.utils import load_chat_model


async def profile_builder(state: State) -> Dict[str, Any]:
    """Call the LLM with the profile builder prompt to extract user profile information.

    This function prepares the prompt using the PROFILE_BUILDER_SYSTEM_PROMPT, 
    initializes the model, and processes the response.
    If the model's response contains extraction data in JSON format, it will be parsed
    and added to the state.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message and any extracted profile data.
    """
    from vibe_trader_agent.prompts import PROFILE_BUILDER_SYSTEM_PROMPT
    
    configuration = Configuration.from_context()

    # Initialize the model
    model = load_chat_model(configuration.model)

    # Use the PROFILE_BUILDER_SYSTEM_PROMPT
    system_message = PROFILE_BUILDER_SYSTEM_PROMPT

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Prepare the result with the response message
    result: Dict[str, Any] = {"messages": [response]}
    
    # Check if the response contains the extraction completion marker
    if isinstance(response.content, str) and "EXTRACTION COMPLETE" in response.content:
        # Try to extract the JSON data
        try:
            # Find the JSON block between ```json and ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                extracted_data = json.loads(json_str)
                
                # Extract profile information
                if "name" in extracted_data:
                    result["name"] = extracted_data["name"]
                
                if "age" in extracted_data:
                    result["age"] = extracted_data["age"]
                
                if "start_portfolio" in extracted_data:
                    result["start_portfolio"] = extracted_data["start_portfolio"]
                
                if "planning_horizon" in extracted_data:
                    result["planning_horizon"] = extracted_data["planning_horizon"]
                
                if "maximum_drawdown_percentage" in extracted_data:
                    result["maximum_drawdown_percentage"] = extracted_data["maximum_drawdown_percentage"]
                
                if "worst_day_decline_percentage" in extracted_data:
                    result["worst_day_decline_percentage"] = extracted_data["worst_day_decline_percentage"]
                
                if "cash_reserve" in extracted_data:
                    result["cash_reserve"] = extracted_data["cash_reserve"]
                
                if "max_single_asset_allocation_percentage" in extracted_data:
                    result["max_single_asset_allocation_percentage"] = extracted_data["max_single_asset_allocation_percentage"]
                
                if "target_amount" in extracted_data:
                    result["target_amount"] = extracted_data["target_amount"]
                
                # Add routing signal to hand off to financial advisor
                result["next"] = "financial_advisor"
                                
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, just continue without updating state
            pass
            
    # Return the model's response and any extracted data
    return result


async def financial_advisor(state: State) -> Dict[str, Any]:
    """Call the LLM with the financial advisor prompt to extract investment preferences.

    This function prepares the prompt using the CONSTRAINTS_EXTRACTOR_SYSTEM_PROMPT,
    initializes the model, and processes the response.
    If the model's response contains extraction data in JSON format, it will be parsed
    and added to the state.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message and any extracted investment data.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding
    model = load_chat_model(configuration.model).bind_tools(financial_advisor_tools)

    # Format the system prompt with current time
    system_message = CONSTRAINTS_EXTRACTOR_SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Prepare the result with the response message
    result: Dict[str, Any] = {"messages": [response]}
    
    # Check if the response contains the extraction completion marker
    if isinstance(response.content, str) and "EXTRACTION COMPLETE" in response.content:
        # Try to extract the JSON data
        try:
            # Find the JSON block between ```json and ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                extracted_data = json.loads(json_str)
                
                # Update the state with extracted data
                if "existing_holdings" in extracted_data and extracted_data["existing_holdings"]:
                    result["existing_holdings"] = extracted_data["existing_holdings"]
                
                if "excluded_assets" in extracted_data and extracted_data["excluded_assets"]:
                    result["excluded_assets"] = extracted_data["excluded_assets"]
                
                if "investment_preferences" in extracted_data and extracted_data["investment_preferences"]:
                    result["investment_preferences"] = extracted_data["investment_preferences"]

                result["next"] = "world_discovery"

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, just continue without updating state
            pass
            
    # Return the model's response and any extracted data
    return result


async def world_discovery(state: State) -> Dict[str, Any]:
    """Call the LLM with the financial advisor prompt to extract investment preferences.

    This function prepares the prompt using the CONSTRAINTS_EXTRACTOR_SYSTEM_PROMPT,
    initializes the model, and processes the response.
    If the model's response contains extraction data in JSON format, it will be parsed
    and added to the state.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message and any extracted investment data.
    """
    configuration = Configuration.from_context()

    model = load_chat_model(configuration.model).bind_tools(world_discovery_tools)
    system_message = WORLD_DISCOVERY_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
        )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    result: Dict[str, Any] = {"messages": [response]}

    if isinstance(response.content, str) and "EXTRACTION COMPLETE" in response.content:
        # Try to extract the JSON data
        try:
            # Find the JSON block between ```json and ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                extracted_data = json.loads(json_str)

                if "tickers_world" in extracted_data:
                    result["tickers_world"] = extracted_data["tickers_world"]
                    result["next"] = "views_analyst"

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, just continue without updating state
            pass

    return result


async def views_analyst(state: State) -> Dict[str, Any]:
    """Call the LLM with the views analyst prompt to generate data structures for Black-Litterman model.

    This function prepares the prompt using the VIEWS_ANALYST_SYSTEM_PROMPT,
    initializes the reasoning model, and processes the response.
    If the model's response contains data in JSON format, it will be parsed
    and added to the state.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Create a reasoning model
    reasoning = {
        "effort": "medium",  # 'low', 'medium', or 'high'
        "summary": None,     # 'detailed', 'auto', or None
    }
    reason_model = ChatOpenAI(
        model="o3-mini",
        use_responses_api=True,
        model_kwargs={"reasoning": reasoning}
    )

    # Initialize the model with tool binding
    model = reason_model.bind_tools(views_analyst_tools)

    # Format the system prompt with current time
    system_message = VIEWS_ANALYST_SYSTEM_PROMPT.format(
        system_time=get_current_date()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=f"List of asset tickers: {state.tickers_world}"),
                *state.messages,
            ]
        ),
    )
    # Note: Only works with `messages` (not State.tickers)

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Prepare the result with the response message
    result: Dict[str, Any] = {"messages": [response]}
    
    # Check if the response contains the extraction completion marker
    if isinstance(response.content, str) and "EXTRACTION COMPLETE".lower() in response.content.lower():        
        extracted_data = extract_json(response.content)
        result["views_created"] = extracted_data if extracted_data else {"error": "missing generated views"}
        
    return result


def human_input_node(state: State) -> Dict[str, Any]:
    """Node that pauses execution to get input from the human user.
    
    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the user's response message.
    """
    # Last msg from the model
    model_response = state.messages[-1].content

    # Use interrupt to pause graph execution and wait for user input
    user_response = interrupt(model_response)
    
    # Return the user input to update the state
    return {"user_input": user_response}
