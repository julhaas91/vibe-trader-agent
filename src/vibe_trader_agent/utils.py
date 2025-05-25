"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def concatenate_mandate_data(existing_holdings, excluded_assets, investment_preferences):
    """Concatenate user-collected madate data into a single string."""
    parts = []
    
    if existing_holdings:
        holdings = []
        for h in existing_holdings:
            text = f"{h['ticker_name']} of quantity {h['quantity']}"
            if h.get('exchange'): text += f" on {h['exchange']}"
            if h.get('region'): text += f" in {h['region']}"
            holdings.append(text)
        parts.append(f"existing_holdings: {', '.join(holdings)}")
    
    if excluded_assets:
        excluded = []
        for e in excluded_assets:
            text = f"{e['ticker_name']} (reason: {e['reason']})"
            if e.get('exchange'): text += f" on {e['exchange']}"
            if e.get('region'): text += f" in {e['region']}"
            excluded.append(text)
        parts.append(f"excluded_assets: {', '.join(excluded)}")
    
    if investment_preferences:
        prefs = ", ".join([f"{p['preference_type']}: {p['description']}" for p in investment_preferences])
        parts.append(f"investment_preferences: {prefs}")
    
    return "; ".join(parts)
