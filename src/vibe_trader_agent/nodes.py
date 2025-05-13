"""Node module for the Vibe Trader Agent."""

import json
import re
from datetime import datetime, UTC
from typing import Any, Dict, cast, Literal

from langchain_core.messages import AIMessage

from vibe_trader_agent.tools import TOOLS

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.utils import load_chat_model
from vibe_trader_agent.state import State
from vibe_trader_agent.prompts import (
    CONSTRAINTS_EXTRACTOR_SYSTEM_PROMPT,
    WORLD_DISCOVERY_PROMPT
    )


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
    result = {"messages": [response]}
    
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
                                
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, just continue without updating state
            print(f"Failed to parse extraction data: {e}")
            
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
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

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
    result = {"messages": [response]}
    
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
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, just continue without updating state
            print(f"Failed to parse extraction data: {e}")
            
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

    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = WORLD_DISCOVERY_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
        )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    result = {"messages": [response]}

    if isinstance(response.content, str) and "EXTRACTION COMPLETE" in response.content:
        # Try to extract the JSON data
        try:
            # Find the JSON block between ```json and ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                extracted_data = json.loads(json_str)

                if "tickers" in extracted_data:
                    result["tickers"] = extracted_data["tickers"]
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, just continue without updating state
            print(f"Failed to parse extraction data: {e}")

    return result



def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"
