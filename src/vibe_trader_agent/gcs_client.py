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
        # Use environment variable if project_id not provided
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("Project ID required via parameter or GOOGLE_CLOUD_PROJECT env var")
        
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and os.path.exists(credentials_path):
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            return storage.Client(project=project_id, credentials=credentials)
        
        return storage.Client(project=project_id)
    
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

