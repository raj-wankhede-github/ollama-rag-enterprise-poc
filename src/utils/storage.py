"""
File storage abstraction layer supporting local storage.
"""

import os
import shutil
from abc import ABC, abstractmethod
from typing import List
from .logger import get_logger

logger = get_logger(__name__)


class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    def upload_file(self, file_path: str, key: str) -> bool:
        """Upload a file to storage"""
        pass
    
    @abstractmethod
    def download_file(self, key: str, local_path: str) -> bool:
        """Download a file from storage"""
        pass
    
    @abstractmethod
    def delete_file(self, key: str) -> bool:
        """Delete a file from storage"""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage with optional prefix"""
        pass
    
    @abstractmethod
    def file_exists(self, key: str) -> bool:
        """Check if file exists in storage"""
        pass


class LocalStorage(StorageProvider):
    """Local file system storage provider"""
    
    def __init__(self, base_dir: str = "./data/uploads"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def upload_file(self, file_path: str, key: str) -> bool:
        """Copy file to local storage directory"""
        try:
            full_path = os.path.join(self.base_dir, key)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as src:
                    with open(full_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                logger.info(f"File uploaded locally: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error uploading file locally: {e}")
            return False
    
    def download_file(self, key: str, local_path: str) -> bool:
        """Read file from local storage"""
        try:
            full_path = os.path.join(self.base_dir, key)
            if os.path.exists(full_path):
                with open(full_path, 'rb') as src:
                    with open(local_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                logger.info(f"File downloaded locally: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error downloading file locally: {e}")
            return False
    
    def delete_file(self, key: str) -> bool:
        """Delete file from local storage"""
        try:
            full_path = os.path.join(self.base_dir, key)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"File deleted locally: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file locally: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in local storage with optional prefix"""
        try:
            files = []
            search_dir = os.path.join(self.base_dir, prefix)
            if os.path.exists(search_dir):
                for root, dirs, filenames in os.walk(search_dir):
                    for filename in filenames:
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, self.base_dir)
                        files.append(rel_path)
            return files
        except Exception as e:
            logger.error(f"Error listing files locally: {e}")
            return []
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists in local storage"""
        full_path = os.path.join(self.base_dir, key)
        return os.path.exists(full_path)


def get_storage_provider(storage_type: str, **kwargs) -> StorageProvider:
    """Factory function to get appropriate storage provider"""
    return LocalStorage(base_dir=kwargs.get("base_dir", "./data/uploads"))
