"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Optional, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch  # type: ignore[import-untyped]

from vibe_trader_agent.configuration import Configuration
from vibe_trader_agent.finance_tools import calculate_financial_metrics


@tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


@tool
async def lookup_financial_info(query: str) -> Optional[dict[str, Any]]:
    """Look up financial information about a company, ticker symbol, or financial concept.
    
    This function performs a specialized search for financial information using the Tavily
    search engine. It's particularly useful for:
    - Looking up ticker symbols for companies
    - Finding information about financial concepts (ESG, market cap, etc.)
    - Getting details about specific assets or investment vehicles
    
    Args:
        query: The search query, which can be a company name, ticker symbol, or financial concept
        
    Returns:
        A dictionary containing search results with relevant financial information
    """
    configuration = Configuration.from_context()
    # Add financial context to the search query
    enhanced_query = f"financial information {query} ticker symbol stock market"
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": enhanced_query}))


@tool
async def search_market_data(query: str) -> Optional[dict[str, Any]]:
    """Search financial markets, news, and company data using Tavily.
    
    Use this tool to find current market information, analyst forecasts,
    earnings guidance, sector trends, and financial news for specific tickers.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# Define individual tools per each node 
financial_advisor_tools = [search]
world_discovery_tools = [search]
views_analyst_tools = [search_market_data, calculate_financial_metrics]
