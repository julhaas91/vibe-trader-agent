"""Utils module for helper functions."""

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, Union

from vibe_trader_agent.state import State


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


def save_state_to_json(state: State, filepath: str = "./user_state.json") -> None:
    """Save State object to JSON file.
    
    Args:
        state: State dataclass instance
        filepath: Path to save the JSON file
    """
    try:
        # Convert dataclass to dictionary
        state_dict = asdict(state)
        
        # Add metadata
        state_dict['_metadata'] = {
            'saved_at': get_current_date(),
            'class_name': state.__class__.__name__,
            'version': '1.0'
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False, default=str)
            
    except Exception:
        pass
