"""Test the JSON extraction functionality for investment data."""

import json
import re


def test_json_extraction_regex():
    """Test the regex pattern used to extract JSON from model responses."""
    # Sample response that would come from the model
    mock_content = """
Thank you for sharing your investment preferences. Here's a summary:

EXTRACTION COMPLETE
```json
{
    "existing_holdings": [
        {"ticker_name": "AAPL", "quantity": 10, "exchange": "NASDAQ", "region": "US"}
    ],
    "excluded_assets": [
        {"ticker_name": "Oil", "reason": "Environmental concerns", "exchange": null, "region": null}
    ],
    "investment_preferences": [
        {"preference_type": "sector", "description": "Renewable energy"}
    ]
}
```
    """
    
    # Use the same regex pattern as in the graph.py file
    json_match = re.search(r"```json\s*(.*?)\s*```", mock_content, re.DOTALL)
    assert json_match is not None, "JSON pattern not found in content"
    
    # Extract the JSON string and parse it
    json_str = json_match.group(1)
    data = json.loads(json_str)
    
    # Verify the extracted data
    assert "existing_holdings" in data
    assert "excluded_assets" in data
    assert "investment_preferences" in data
    
    # Check specific values
    assert data["existing_holdings"][0]["ticker_name"] == "AAPL"
    assert data["existing_holdings"][0]["quantity"] == 10
    assert data["excluded_assets"][0]["reason"] == "Environmental concerns"
    assert data["investment_preferences"][0]["preference_type"] == "sector"
    assert data["investment_preferences"][0]["description"] == "Renewable energy"


def test_json_extraction_with_empty_arrays():
    """Test extraction of JSON with empty arrays."""
    # Sample response with empty arrays
    mock_content = """
EXTRACTION COMPLETE
```json
{
    "existing_holdings": [],
    "excluded_assets": [],
    "investment_preferences": [
        {"preference_type": "sector", "description": "Renewable energy"}
    ]
}
```
    """
    
    # Find and extract the JSON
    json_match = re.search(r"```json\s*(.*?)\s*```", mock_content, re.DOTALL)
    assert json_match is not None
    
    json_str = json_match.group(1)
    data = json.loads(json_str)
    
    # Check that empty arrays are properly handled
    assert len(data["existing_holdings"]) == 0
    assert len(data["excluded_assets"]) == 0
    assert len(data["investment_preferences"]) == 1