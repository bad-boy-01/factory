import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DriveUploader:
    """
    Utility to upload files to Google Drive using a Service Account.
    Optimized for Kaggle environment using Kaggle Secrets.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.service = None
        self._authenticated = False
        
    def _authenticate(self) -> bool:
        if self._authenticated:
            return True
            
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            # 1. Try to get JSON from Kaggle Secrets
            service_account_info = None
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                json_str = user_secrets.get_secret("GDRIVE_SERVICE_ACCOUNT_JSON")
                if json_str:
                    service_account_info = json.loads(json_str)
                    logger.info("Found Google Drive credentials in Kaggle Secrets.")
            except Exception:
                # Not in Kaggle or secret missing
                pass
                
            # 2. Fallback to local file if config provides one
            if not service_account_info:
                creds_path = self.config.get('gdrive', {}).get('creds_path')
                if creds_path and os.path.exists(creds_path):
                    with open(creds_path, 'r') as f:
                        service_account_info = json.load(f)
                    logger.info(f"Loaded Google Drive credentials from {creds_path}")

            if not service_account_info:
                logger.warning("Google Drive credentials not found. Skipping upload. (Set 'GDRIVE_SERVICE_ACCOUNT_JSON' in Kaggle Secrets)")
                return False
                
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES
            )
            self.service = build('drive', 'v3', credentials=creds)
            self._authenticated = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Drive: {e}")
            return False

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> Optional[str]:
        """Uploads a file and returns its Drive ID."""
        if not self._authenticate():
            return None
            
        if not os.path.exists(file_path):
            logger.error(f"File not found for upload: {file_path}")
            return None
            
        try:
            from googleapiclient.http import MediaFileUpload
            
            file_name = os.path.basename(file_path)
            file_metadata = {'name': file_name}
            
            target_folder = folder_id or self.config.get('gdrive', {}).get('folder_id')
            if target_folder:
                file_metadata['parents'] = [target_folder]

            media = MediaFileUpload(file_path, resumable=True)
            
            logger.info(f"Uploading {file_name} to Google Drive...")
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            drive_id = file.get('id')
            logger.info(f"Successfully uploaded to Drive! File ID: {drive_id}")
            return drive_id
            
        except Exception as e:
            logger.error(f"Google Drive upload failed: {e}")
            return None
