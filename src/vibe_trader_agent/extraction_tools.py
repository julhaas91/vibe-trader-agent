"""Structured Output as extraction tools."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain_core.tools import tool

from vibe_trader_agent.finance_tools import validate_ticker_exists


@tool
def extract_profile_data(
    name: str = "",
    age: int = 0,
    start_portfolio: float = 0.0,
    planning_horizon: str = "",
    maximum_drawdown_percentage: float = 0.0,
    worst_day_decline_percentage: float = 0.0,
    cash_reserve: float = 0.0,
    max_single_asset_allocation_percentage: float = 0.0,
    target_amount: float = 0.0
) -> Dict[str, Any]:
    """Extract and confirm user profile information when ALL required fields are collected.

    Args:
        name: User's full name
        age: User's age in years
        start_portfolio: Initial investment capital available
        planning_horizon: Investment timeframe (e.g., "10 years", "5 months")
        maximum_drawdown_percentage: Maximum acceptable portfolio decline (%)
        worst_day_decline_percentage: Maximum tolerable single-day loss (%)
        cash_reserve: Amount to keep liquid for emergencies
        max_single_asset_allocation_percentage: Maximum allocation to single investment (%)
        target_amount: Financial goal or target portfolio value
    """
    return {
        "name": name,
        "age": age,
        "start_portfolio": start_portfolio,
        "planning_horizon": planning_horizon,
        "maximum_drawdown_percentage": maximum_drawdown_percentage,
        "worst_day_decline_percentage": worst_day_decline_percentage,
        "cash_reserve": cash_reserve,
        "max_single_asset_allocation_percentage": max_single_asset_allocation_percentage,
        "target_amount": target_amount,
        "profile_complete": True,
    }


@tool
def extract_mandate_data(
    existing_holdings: Optional[List[Dict[str, Any]]] = [],
    excluded_assets: Optional[List[Dict[str, Any]]] = [],
    investment_preferences: Optional[List[Dict[str, Any]]] = []
) -> Dict[str, Any]:
    """Extract and confirm user investment information including holdings, exclusions, and preferences.
    
    Args:
        existing_holdings: List of current asset holdings with ticker, quantity, and optional exchange/region
        excluded_assets: List of assets to exclude with ticker/category, reason, and optional exchange/region  
        investment_preferences: List of investment preferences with type and description
        
    Returns:
        Dict containing validated investment data with structure:
        {
            "existing_holdings": [{"ticker_name": str, "quantity": float, "exchange": str, "region": str}],
            "excluded_assets": [{"ticker_name": str, "reason": str, "exchange": str, "region": str}],
            "investment_preferences": [{"preference_type": str, "description": str}]
        }
    """
    return {
        "existing_holdings": existing_holdings,
        "excluded_assets": excluded_assets, 
        "investment_preferences": investment_preferences,
        "mandate_complete": True,
    }


@tool
def extract_tickers_data(
    tickers: List[str]
) -> Dict[str, Union[List[str], str]]:
    """Extract and confirm ticker symbols for investments.
    
    Args:
        tickers: List of ticker symbols. Should contain at least two valid tickers.
    
    Returns:
        Dict containing validated ticker data with structure:
        {
            "tickers": ["TICKER_1", "TICKER_2", "TICKER_3", ...]
        }
        OR on failure:
        {
            "error": "Insufficient valid tickers found. Need at least two valid."
        }
    """
    
    def validate_tickers(ticker_list: List[str]) -> List[str]:
        """Validate and normalize ticker symbols."""
        validated_tickers = []
        
        for ticker in ticker_list:
            # Handle non-string inputs
            if not isinstance(ticker, str):
                continue
            
            # Clean and normalize ticker
            cleaned_ticker = ticker.strip().upper()
            
            # Skip empty tickers
            if not cleaned_ticker:
                continue

            if validate_ticker_exists(cleaned_ticker):
                validated_tickers.append(cleaned_ticker)
        
        # Remove duplicates
        return list(set(validated_tickers))
    
    # Validate and process tickers
    valid_tickers = validate_tickers(tickers)
    
    # Return tickers if at least 2 valid
    if len(valid_tickers) >= 2:
        return {
            "tickers": valid_tickers
        }
    
    # Return error if insufficient valid tickers
    return {
        "error": f"Insufficient valid tickers found. Need at least 2, got {len(valid_tickers)} - {valid_tickers}. Please research additional assets for portfolio."
    }


@tool
def extract_bl_views(
    views: List[Dict[str, Any]] = []
) -> Dict[str, Any]:
    """Extract and confirm Black-Litterman views for portfolio optimization.
    
    Args:
        views: List of investment views containing absolute or relative return expectations.
               Each view should have structure:
               - For absolute views: {
                   "view_type": "absolute",
                   "ticker": str,
                   "expected_return": float,
                   "uncertainty": float,
                   "description": str
                 }
               - For relative views: {
                   "view_type": "relative", 
                   "long_ticker": str,
                   "short_ticker": str,
                   "expected_return": float,
                   "uncertainty": float,
                   "description": str
                 }
    
    Returns:
        Dict containing processed Black-Litterman matrices.
    """
    try:

        # Validate views structure and content
        is_valid, error_message = _validate_bl_views(views)
        if not is_valid:
            return {
                "error": error_message,
                "views_complete": False
            }
        
        # Process views into the structured nested format
        result = _process_bl_views(views)
        
        # Add completion flag
        result["views_complete"] = True
        
        return {
            "bl_views": result
        }
    
    except ValueError as e:
        # Catch specific validation errors from process_bl_views
        return {
            "error": f"Processing error: {str(e)}",
            "views_complete": False
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "views_complete": False
        }


def _validate_bl_views(views: List[Dict[str, Any]]) -> tuple[bool, str]:
    """Validate Black-Litterman views structure and content.
    
    Args:
        views: List of view dictionaries to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        If valid, error_message will be empty string
    """
    # Validate input exists and is a list
    if not views:
        return False, "No views provided. At least one view is required."
    
    if not isinstance(views, list):
        return False, f"Views must be a list, got {type(views).__name__}."
    
    # Validate each view structure
    for i, view in enumerate(views):
        if not isinstance(view, dict):
            return False, f"View {i+1} must be a dictionary, got {type(view).__name__}."
        
        # Check view_type
        if "view_type" not in view:
            return False, f"View {i+1} is missing required field 'view_type'."
        
        view_type = view.get("view_type")
        if view_type not in ["absolute", "relative"]:
            return False, f"View {i+1} has invalid view_type '{view_type}'. Must be 'absolute' or 'relative'."
        
        # Check common required fields
        common_fields = ["expected_return", "uncertainty", "description"]
        for field in common_fields:
            if field not in view:
                return False, f"View {i+1} is missing required field '{field}'."
        
        # Check type-specific required fields
        if view_type == "absolute":
            if "ticker" not in view:
                return False, f"View {i+1} (absolute) is missing required field 'ticker'."
            if not isinstance(view["ticker"], str) or not view["ticker"].strip():
                return False, f"View {i+1} has invalid ticker: must be a non-empty string."
        else:  # relative
            missing_fields = []
            if "long_ticker" not in view:
                missing_fields.append("long_ticker")
            if "short_ticker" not in view:
                missing_fields.append("short_ticker")
            
            if missing_fields:
                return False, f"View {i+1} (relative) is missing required field(s): {', '.join(missing_fields)}."
            
            # Validate ticker values
            if not isinstance(view["long_ticker"], str) or not view["long_ticker"].strip():
                return False, f"View {i+1} has invalid long_ticker: must be a non-empty string."
            if not isinstance(view["short_ticker"], str) or not view["short_ticker"].strip():
                return False, f"View {i+1} has invalid short_ticker: must be a non-empty string."
            if view["long_ticker"] == view["short_ticker"]:
                return False, f"View {i+1} has same ticker '{view['long_ticker']}' for both long_ticker and short_ticker."
        
        # Validate numeric fields
        try:
            expected_return = float(view["expected_return"])
        except (ValueError, TypeError):
            return False, f"View {i+1} has invalid expected_return: must be a number, got '{view['expected_return']}'."
        
        try:
            uncertainty = float(view["uncertainty"])
        except (ValueError, TypeError):
            return False, f"View {i+1} has invalid uncertainty: must be a number, got '{view['uncertainty']}'."
        
        # Validate ranges
        if not (-1.0 <= expected_return <= 1.0):
            return False, f"View {i+1} expected_return {expected_return} is outside reasonable range [-1.0, 1.0]."
        
        if not (0.0001 <= uncertainty <= 0.1):
            return False, f"View {i+1} uncertainty {uncertainty} is outside valid range [0.0001, 0.1]."
        
        # Validate description
        if not isinstance(view["description"], str) or not view["description"].strip():
            return False, f"View {i+1} has invalid description: must be a non-empty string."
    
    return True, ""


def _process_bl_views(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert structured views (generated by LLM) to Black-Litterman parameters.
    
    Args:
        views: List of view dictionaries
    
    Returns:
        Dict with p_matrix, q_vector, sigma_vector, explanation, tickers
    """
    # Validate basic structure
    if not isinstance(views, list) or len(views) == 0:
        raise ValueError("Views must be a non-empty list")
    
    # Extract all unique tickers from views in deterministic order
    tickers_set = set()
    for view in views:
        if view.get("view_type") == "absolute":
            if "ticker" in view:
                tickers_set.add(view["ticker"])
        elif view.get("view_type") == "relative":
            if "long_ticker" in view:
                tickers_set.add(view["long_ticker"])
            if "short_ticker" in view:
                tickers_set.add(view["short_ticker"])
    
    # Sort tickers for consistent ordering
    tickers = sorted(list(tickers_set))
    
    # Create ticker index mapping
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    k = len(tickers)  # number of assets
    v = len(views)    # number of views
    
    # Initialize matrices
    p_matrix = np.zeros((v, k))
    q_vector = np.zeros(v)
    sigma_vector = np.zeros(v)
    descriptions = []
    
    # Process each view
    for i, view in enumerate(views):
        # Validate required fields
        required_fields = ["view_type", "expected_return", "uncertainty", "description"]
        for field in required_fields:
            if field not in view:
                raise ValueError(f"View {i+1}: Missing field '{field}'")
        
        # Extract common fields
        view_type = view["view_type"]
        expected_return = float(view["expected_return"])
        uncertainty = float(view["uncertainty"])
        description = view["description"]
        
        # Validate uncertainty range
        if not (0.0001 <= uncertainty <= 0.1):
            raise ValueError(f"View {i+1}: Uncertainty {uncertainty} outside valid range [0.0001, 0.1]")
        
        # Process by view type
        if view_type == "absolute":
            if "ticker" not in view:
                raise ValueError(f"View {i+1}: Absolute view missing 'ticker'")
            
            ticker = view["ticker"]
            if ticker not in ticker_to_idx:
                raise ValueError(f"View {i+1}: Unknown ticker '{ticker}'")
            
            # Set P matrix: 1 for the target ticker
            p_matrix[i, ticker_to_idx[ticker]] = 1
            descriptions.append(f"{ticker} with expected return {expected_return:.1%} due to {description}.")
            
        elif view_type == "relative":
            if "long_ticker" not in view or "short_ticker" not in view:
                raise ValueError(f"View {i+1}: Relative view missing ticker fields")
            
            long_ticker = view["long_ticker"]
            short_ticker = view["short_ticker"]
            
            if long_ticker not in ticker_to_idx:
                raise ValueError(f"View {i+1}: Unknown long_ticker '{long_ticker}'")
            if short_ticker not in ticker_to_idx:
                raise ValueError(f"View {i+1}: Unknown short_ticker '{short_ticker}'")
            if long_ticker == short_ticker:
                raise ValueError(f"View {i+1}: Cannot have same ticker for long and short")
            
            # Set P matrix: 1 for long, -1 for short
            p_matrix[i, ticker_to_idx[long_ticker]] = 1
            p_matrix[i, ticker_to_idx[short_ticker]] = -1
            descriptions.append(f"{long_ticker} outperforms {short_ticker} by {expected_return:.1%} due to {description}.")
            
        else:
            raise ValueError(f"View {i+1}: Invalid view_type '{view_type}'. Must be 'absolute' or 'relative'")
        
        # Set q vector and sigma vector
        q_vector[i] = expected_return
        sigma_vector[i] = uncertainty
    
    return {
        "p_matrix": p_matrix.tolist(),
        "q_vector": q_vector.tolist(), 
        "sigma_vector": sigma_vector.tolist(),
        "explanation": "; ".join(descriptions),
        "tickers": tickers  # Include the ticker ordering for reference
    }
