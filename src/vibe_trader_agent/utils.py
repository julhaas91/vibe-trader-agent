"""Utility & helper functions."""
import os
import datetime as dt
from io import BytesIO
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from google.cloud import storage


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


def concatenate_mandate_data(
    existing_holdings: Optional[List[Dict[str, Any]]], 
    excluded_assets: Optional[List[Dict[str, Any]]], 
    investment_preferences: Optional[List[Dict[str, Any]]]
) -> str:
    """Concatenate user-collected mandate data into a single string.
    
    Args:
        existing_holdings: List of holdings dictionaries containing ticker_name, 
            quantity, and optionally exchange and region keys.
        excluded_assets: List of excluded asset dictionaries containing ticker_name, 
            reason, and optionally exchange and region keys.
        investment_preferences: List of preference dictionaries containing 
            preference_type and description keys.
    
    Returns:
        Semicolon-separated string of formatted mandate data, or empty string 
        if all inputs are None/empty.    
    """        
    parts = []
    
    if existing_holdings:
        holdings = []
        for h in existing_holdings:
            text = f"{h['ticker_name']} of quantity {h['quantity']}"
            if h.get('exchange'):
                text += f" on {h['exchange']}"
            if h.get('region'):
                text += f" in {h['region']}"
            holdings.append(text)
        parts.append(f"existing_holdings: {', '.join(holdings)}")
    
    if excluded_assets:
        excluded = []
        for e in excluded_assets:
            text = f"{e['ticker_name']} (reason: {e['reason']})"
            if e.get('exchange'):
                text += f" on {e['exchange']}"
            if e.get('region'):
                text += f" in {e['region']}"
            excluded.append(text)
        parts.append(f"excluded_assets: {', '.join(excluded)}")
    
    if investment_preferences:
        prefs = ", ".join([f"{p['preference_type']}: {p['description']}" for p in investment_preferences])
        parts.append(f"investment_preferences: {prefs}")
    
    return "; ".join(parts)


class GCStorage:
    def __init__(self, storage_client: storage.Client) -> None:
        self.client = storage_client

    def get_bucket(self, bucket_name: str) -> storage.Bucket:
        return self.client.get_bucket(bucket_name)

    def upload_bytes(
            self,
            bucket: storage.Bucket,
            destination: str,
            content: bytes,
            public: bool,
            expiration_days: int = 7
            ) -> str:
        blob = bucket.blob(destination)
        blob.upload_from_string(content, content_type='application/pdf')

        if public:
            return blob.generate_signed_url(
                version="v4",
                expiration=dt.timedelta(days=expiration_days),
                method="GET"
                )
        else:
            return f"gs://{bucket.name}/{destination}"


def init_storage_client(project_id: str) -> storage.Client:
    try:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
                )
            return storage.Client(project=project_id, credentials=credentials)
        return storage.Client(project=project_id)
    except Exception as e:
        raise OSError(
            "Failed to initialize storage client. Ensure GOOGLE_CLOUD_PROJECT "
            "environment variable is set or pass project_id explicitly."
            ) from e


def upload_pdf(client: storage.Client, bucket: str, destination: str, content: bytes, make_public: bool = False, expiration_days: int = 7) -> None | str:

    gcs = GCStorage(client)
    bucket_gcs = gcs.get_bucket(bucket)
    return gcs.upload_bytes(
        bucket=bucket_gcs,
        destination=destination,
        content=content,
        public=make_public,
        expiration_days=expiration_days
        )
