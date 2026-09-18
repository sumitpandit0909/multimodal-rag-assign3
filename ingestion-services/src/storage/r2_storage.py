import os
import mimetypes
import logging
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    Config = None

logger = logging.getLogger(__name__)

class R2StorageManager:
    """
    Cloudflare R2 S3-compatible object storage manager.
    Handles high-speed, zero-egress cloud storage for page screenshots and processed PDFs.
    """
    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "").strip()
        self.public_url = os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")
        
        self._s3_client = None
        if self.is_configured() and boto3:
            self._init_client()

    def is_configured(self) -> bool:
        """Returns True if all required Cloudflare R2 credentials are set."""
        return bool(
            self.account_id and 
            self.access_key_id and 
            self.secret_access_key and 
            self.bucket_name and 
            boto3
        )

    def _init_client(self):
        """Initializes the S3 client configured for Cloudflare R2 endpoint."""
        endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        try:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=30
                ),
                region_name="auto"
            )
            logger.info(f"[Cloudflare R2] S3 client initialized for bucket '{self.bucket_name}' at {endpoint_url}")
        except Exception as e:
            logger.error(f"[Cloudflare R2] Failed to initialize S3 client: {e}")
            self._s3_client = None

    def upload_file(self, file_path: Path, object_key: str, content_type: Optional[str] = None) -> Optional[str]:
        """
        Uploads a local file to the Cloudflare R2 bucket.
        Returns the public CDN / direct URL, or None if upload fails.
        """
        if not self.is_configured() or not self._s3_client:
            logger.debug(f"[Cloudflare R2] Not configured. Skipping upload for {file_path.name}")
            return None

        if not file_path.exists():
            logger.error(f"[Cloudflare R2] File not found: {file_path}")
            return None

        # Determine MIME type if not specified
        if not content_type:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if not content_type:
                content_type = "image/png" if file_path.suffix.lower() == ".png" else "application/octet-stream"

        try:
            extra_args = {"ContentType": content_type}
            logger.info(f"[Cloudflare R2] Uploading {file_path.name} -> s3://{self.bucket_name}/{object_key} ({content_type})...")
            
            self._s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.bucket_name,
                Key=object_key,
                ExtraArgs=extra_args
            )

            # Build URL
            if self.public_url:
                public_file_url = f"{self.public_url}/{object_key}"
            else:
                public_file_url = f"https://{self.account_id}.r2.cloudflarestorage.com/{self.bucket_name}/{object_key}"

            logger.info(f"[Cloudflare R2] Successfully uploaded. URL: {public_file_url}")
            return public_file_url

        except Exception as e:
            logger.error(f"[Cloudflare R2] Upload failed for {file_path.name}: {e}")
            return None


# Global singleton instance
r2_storage = R2StorageManager()
