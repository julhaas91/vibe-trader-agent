"""This module defines tools for nodes in VibeTrader."""

from typing import Any, Optional, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch  # type: ignore[import-untyped]

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.finance_tools import calculate_financial_metrics
from vibe_trader_agent.extraction_tools import (
    extract_profile_data,
    extract_mandate_data,
    extract_tickers_data,
    extract_bl_views,
)


@tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search the web for current information, facts, and answers to questions.

    Use this tool to find up-to-date information from the internet when you need:
    - Current events, news, or recent developments
    - Real-time data (stock prices, exchange rates, world events etc.)
    - Factual information you are uncertain about
    - Answers to specific questions you cannot answer from your knowledge
    - Verification of claims or statements
    - Recent changes in policies, regulations, or procedures
    
    Args:
        query (str): Your search query. Be specific and concise. Examples:
    
    Returns:
        dict: Search results containing relevant information from web sources,
              or None if search fails.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


@tool
async def search_market_data(query: str) -> Optional[dict[str, Any]]:
    """Search financial markets, news, and company data for investment analysis and views generation.

    Use this tool to collect up-to-date data to generate absolute or relative views:
    - Current market sentiments about specific stocks/markets/themes
    - Company earnings reports, financial statements, and SEC filings
    - Analyst forecasts, price targets, and investment recommendations
    - Economic indicators, sector performance, and market trends
    - Breaking financial news and market-moving events

    Args:
        query (str): Your search query. Be specific with ticker symbols and financial terms. 
                    Examples:
                    - "AAPL quarterly earnings Q4 2025"
                    - "Tesla stock price analyst forecast"
                    - "S&P 500 sector rotation trends"
                    - "Federal Reserve interest rate decision impact"
    
    Returns:
        dict: Financial data and news results from market sources,
              or None if search fails.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# Define individual tools per each node 
profile_builder_tools = [search, extract_profile_data]
financial_advisor_tools = [search, extract_mandate_data]
asset_finder_tools = [search, extract_tickers_data]
views_analyst_tools = [search_market_data, calculate_financial_metrics, extract_bl_views]
