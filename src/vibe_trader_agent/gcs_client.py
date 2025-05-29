"""Google Cloud Storage utilities for file operations."""

import os
from datetime import timedelta
from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from vibe_trader_agent.optimization.pdf_dashboard import generate_pdf_dashboard

class GCStorage:
    """Simple Google Cloud Storage wrapper."""
    
    def __init__(self, storage_client: storage.Client) -> None:
        self.client = storage_client

    def get_bucket(self, bucket_name: str) -> storage.Bucket:
        """Get bucket by name."""
        try:
            return self.client.get_bucket(bucket_name)
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to get bucket '{bucket_name}': {e}") from e

    def upload_bytes(
        self,
        bucket: storage.Bucket,
        destination: str,
        content: bytes,
        content_type: str = 'application/pdf',
        public: bool = False,
        expiration_days: int = 7
    ) -> str:
        """Upload bytes to GCS bucket."""
        try:
            blob = bucket.blob(destination)
            blob.upload_from_string(content, content_type=content_type)

            if public:
                return blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(days=expiration_days),
                    method="GET"
                )
            else:
                return f"gs://{bucket.name}/{destination}"
        except GoogleCloudError as e:
            raise RuntimeError(f"Upload failed for '{destination}': {e}") from e


def init_storage_client(project_id: Optional[str] = None) -> storage.Client:
    """Initialize Google Cloud Storage client."""
    try:
        # Get credentials from environment variables
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
        
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return storage.Client(project=project_id, credentials=credentials)
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize storage client: {e}") from e


def upload_pdf(
    bucket_name: str,
    destination: str,
    content: bytes,
    make_public: bool = False,
    expiration_days: int = 7
) -> str:
    """Upload PDF content to Google Cloud Storage."""
    gcs_client = init_storage_client()
    gcs = GCStorage(gcs_client)
    bucket = gcs.get_bucket(bucket_name)
    return gcs.upload_bytes(
        bucket=bucket,
        destination=destination,
        content=content,
        content_type='application/pdf',
        public=make_public,
        expiration_days=expiration_days
    )

