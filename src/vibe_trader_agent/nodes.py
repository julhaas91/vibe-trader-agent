"""Node module for the Vibe Trader Agent."""

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, cast
import asyncio
import httpx
from google.oauth2 import service_account
from google.auth.transport import requests


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.finance_tools import validate_ticker_exists
from vibe_trader_agent.misc import get_current_date, generate_short_id
from vibe_trader_agent.optimization.params_validation import validate_optimizer_params
from vibe_trader_agent.optimization.results_formatting import format_results_for_llm
from vibe_trader_agent.optimization.state_parser import parse_state_to_optimizer_params
from vibe_trader_agent.prompts import (
    ASSET_FINDER_SYSTEM_PROMPT,
    FINANCIAL_ADVISOR_SYSTEM_PROMPT,
    PROFILE_BUILDER_SYSTEM_PROMPT,
    VIEWS_ANALYST_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
)
from vibe_trader_agent.state import State
from vibe_trader_agent.tools import (
    asset_finder_tools,
    financial_advisor_tools,
    profile_builder_tools,
    views_analyst_tools,
)
from vibe_trader_agent.utils import concatenate_mandate_data, load_chat_model
from vibe_trader_agent.gcs_client import upload_pdf
from vibe_trader_agent.optimization.pdf_dashboard import generate_pdf_dashboard


def get_google_credentials():
    """Get Google Cloud credentials for service authentication."""
    service_url = os.getenv("OPTIMIZER_SERVICE_URL")

    credentials_dict = {
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL"),
        "universe_domain": "googleapis.com"
    }
    
    # Validate required fields
    required_fields = ["private_key_id", "private_key", "client_email", "client_id"]
    missing_fields = [field for field in required_fields if not credentials_dict.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    credentials = service_account.IDTokenCredentials.from_service_account_info(
        credentials_dict,
        target_audience=service_url,
        additional_claims={
            "email": credentials_dict["client_email"],
            "email_verified": True
            }
    )

    return credentials


async def profile_builder(state: State) -> Dict[str, Any]:
    """Engage with user to collect profile information and route appropriately.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        dict: Updated state with response and routing information.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tools
    model = load_chat_model(configuration.model)
    model_with_tools = model.bind_tools(profile_builder_tools)

    # Format the system prompt with current time
    system_message = PROFILE_BUILDER_SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(AIMessage, await model_with_tools.ainvoke([
        {"role": "system", "content": system_message},
        *state.messages
    ]))

    # State Update with LLM response
    result: Dict[str, Any] = {}

    # Check if any tools were called by LLM
    if response.tool_calls:
        tool_call = response.tool_calls[0]

        if tool_call["name"] == "extract_profile_data":
            # Profile extraction completed - update state and route to financial advisor
            # Note: don't add tool-call message
            profile_data = tool_call["args"]
            result.update(profile_data)
            result["next"] = "financial_advisor"
            return result
        elif tool_call["name"] == "search":
            # Search tool called - continue to tools node
            result["messages"] = [response]
            result["next"] = "tools_builder"
            return result
    
    # No tools called - continue conversation with user
    result["messages"] = [response]
    result["next"] = "human_input_builder"
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
    system_message = FINANCIAL_ADVISOR_SYSTEM_PROMPT.format(
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

    # State update with LLM response
    result: Dict[str, Any] = {}

    # Check if any tools were called by LLM
    if response.tool_calls:
        tool_call = response.tool_calls[0]

        if tool_call["name"] == "extract_mandate_data":
            # Mandate extraction completed - update state and route to asset finder
            # Note: don't add tool-call message
            mandate_data = tool_call["args"]
            result.update(mandate_data)
            result["next"] = "asset_finder"
            return result
        elif tool_call["name"] == "search":
            # Search tool called - continue to tools node
            result["messages"] = [response]
            result["next"] = "tools_advisor"
            return result

    # No tools called - continue conversation with user
    result["messages"] = [response]
    result["next"] = "human_input_advisor"
    return result


async def asset_finder(state: State) -> Dict[str, Any]:
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

    model = load_chat_model(configuration.model).bind_tools(asset_finder_tools)
    system_message = ASSET_FINDER_SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
        )

    # Merge mandate info together
    user_mandate = concatenate_mandate_data(state.existing_holdings, state.excluded_assets, state.investment_preferences)

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message}, 
                HumanMessage(content=f"My personal structured mandate info:{user_mandate}"),
                *state.messages,
            ]
        ),
    )

    # State Update with LLM response
    result: Dict[str, Any] = {}

    # Check if any tools were called by LLM
    if response.tool_calls:
        tool_call = response.tool_calls[0]

        if tool_call["name"] == "extract_tickers_data":
            # Tickers identified - update state and route to views analyst
            # Note: don't add tool-call message
            tickers = tool_call["args"]            
            tickers["tickers"] = [t for t in tickers["tickers"] if validate_ticker_exists(t)]
            
            result.update(tickers)
            result["next"] = "views_analyst"
            return result
        elif tool_call["name"] == "search":
            # Search tool called - continue to tools node
            result["messages"] = [response]
            result["next"] = "tools_finder"
            return result
    
    # No tools called - continue conversation with user
    result["messages"] = [response]
    result["next"] = "human_input_finder"
    return result


async def views_analyst(state: State) -> Dict[str, Any]:
    """Call the LLM with the views analyst prompt to generate data structures for Black-Litterman model.

    This function first checks if the last message is a ToolMessage from the 'extract_bl_views' tool.
    If so, it returns a success message without calling the LLM.
    Otherwise, it prepares the prompt using the VIEWS_ANALYST_SYSTEM_PROMPT,
    initializes the reasoning model.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Check if the last message is a ToolMessage from extract_bl_views tool
    if state.messages and len(state.messages) > 0:
        last_message = state.messages[-1]
        
        if (isinstance(last_message, ToolMessage) and 
            hasattr(last_message, 'name') and 
            last_message.name == "extract_bl_views" and
            hasattr(last_message, 'tool_call_id') and 
            last_message.tool_call_id):  # Ensure it has a valid tool_call_id

            if isinstance(last_message.content, str):
                tool_output = json.loads(last_message.content)
            else:
                tool_output = last_message.content

            return {
                "messages": [AIMessage(content="Black-Litterman inputs generated and extracted successfully.")],
                "bl_views": tool_output["bl_views"],
                "next": "optimizer"
            }
    
    # Create reasoning model
    reason_model = ChatOpenAI(
        model="o3-mini",            # 'o4-mini'
        reasoning_effort="medium",  # 'low', 'medium', or 'high'
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
                HumanMessage(content=f"List of asset tickers: {state.tickers}"),
                *state.messages,
            ]
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

    # State Update with LLM response
    result: Dict[str, Any] = {"messages": [response]}

    return result


async def human_input_node(state: State) -> Dict[str, Any]:
    """Node that pauses execution to get input from the human user.
    
    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the user's response message.
    """
    # Last msg from the model
    model_response = state.messages[-1].content

    # Use asyncio.to_thread to run the blocking interrupt call in a separate thread
    user_response = await asyncio.to_thread(interrupt, model_response)
    
    # Return the user input to update the state
    return {"user_input": user_response}


async def optimizer(state: State) -> Dict[str, Any]:
    """Node to perform portfolio optimization via Cloud Run service.
    
    Args:
        state (State): The current state of the conversation.

    Returns:
        Dict[str, Any]: LLM-friendly output of the optimization process.
    """
    # Convert State to expected params by Optimizer
    params = parse_state_to_optimizer_params(
        asdict(state), 
        scenarios=5,     # TODO: optimal value
        max_iterations=1,  # TODO: optimal value
        output_dir=None     # No File writing
    )

    try:
        # Todo: fix this and use the environment variable
        # service_url = os.getenv("OPTIMIZER_SERVICE_URL")
        # if not service_url:
        #     raise ValueError("OPTIMIZER_SERVICE_URL environment variable not set")

        # Get Google Cloud credentials for authentication
        credentials = get_google_credentials()

        # Refresh credentials to get access token
        auth_req = requests.Request()
        credentials.refresh(auth_req)
        token = credentials.token

        if not token:
            raise ValueError("Failed to obtain access token")

        # Prepare headers for authenticated request
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Call the Cloud Run service
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            response = await client.post(
                #f"{service_url}/optimize",
                f"https://viber-trader-optimizer-189263797377.europe-west3.run.app/optimize",
                json=params,
                headers=headers
            )
            response.raise_for_status()
            optimizer_results = response.json()

        # Format results into user-friendly message
        formatted_results = format_results_for_llm(optimizer_results)

        return {
            "messages": [AIMessage(content=formatted_results)],
            "optimizer_raw_results": optimizer_results,
            "optimizer_outcome": formatted_results,
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Optimization failed: {str(e)}")],
        }


async def reporter(state: State) -> Dict[str, Any]:
    """Report final results in a user-friendly manner.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        dict: Updated state with response and routing information.
    """
    # Debugging
    # from vibe_trader_agent.misc import save_state_to_json
    # save_state_to_json(state, "./user_state.json")
    # print("SAVED")

    # Debugging    
    # with open("./user_state.json", 'r') as f:
    #     state_loaded = json.load(f)
    # state = State(**state_loaded)

    # Publish PDF Dashboard
    if not state.pdf_dashboard_url:
        try:
            pdf_id = generate_short_id(state.optimizer_outcome)
            pdf_bytes = generate_pdf_dashboard(
                inputs=state.optimizer_raw_results["inputs"], 
                results=state.optimizer_raw_results["results"],
            )
            url = upload_pdf(
                bucket_name="vibe-trader-reports-dev",
                destination=f"vibe_trader_dashboard_{pdf_id}.pdf",
                content=pdf_bytes,
                make_public=True,
                expiration_days=7
            )
        except:
            url = "Invalid URL: Please double check the PDF Dashboard Publication."
    
    configuration = Configuration.from_context()

    # Initialize the model with tools
    model = load_chat_model(configuration.model)

    # Format the system prompt with current time
    system_message = REPORTER_SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(AIMessage, await model.ainvoke([
        {"role": "system", "content": system_message},
        AIMessage(content=f"PDF Dashboard available at: {url}"),
        HumanMessage(content=f"My complete personal data: {asdict(state)}"),
        HumanMessage(content=f"Explain clearly the results of the Portfolio Optimization: {state.optimizer_outcome}. Include link to the PDF Dashboard at the end."),
    ]))
    
    return {"messages": [response]}

