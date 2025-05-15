"""Utils module for helper functions."""

import json
import re
from datetime import UTC, datetime
from typing import Any, Dict, Union


def get_current_date() -> str:
    """Get the current date in UTC timezone.
    
    Returns:
        str: Current date in YYYY-MM-DD format
    """
    return datetime.now(UTC).strftime('%Y-%m-%d')


def extract_json(text: str) -> Union[Dict[Any, Any], Any]:
    """Extract JSON data from a string."""
    try:
        # Look for JSON block between ```json and ```
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Alternative: try to find a JSON object directly in the text
        json_match = re.search(r'\{[\s\S]*?\}', text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
            
        return {}
    except Exception:
        return {}
    
