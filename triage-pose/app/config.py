# Application configuration
"""
Configuration for the Triage-Pose application
"""
from typing import List
from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    app_name: str = "Triage-Pose"
    api_version: str = "0.1.0"
    
    # FastAPI settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # PhysioTrack settings
    physiotrack_version: str = "0.1.0"
    
    # CORS settings
    allowed_origins: List[str] = ["*"]
    
    # Storage settings
    upload_dir: str = "static/uploads"
    results_dir: str = "static/results"
    temp_dir: str = "static/temp"
    
    # Processing settings
    default_model: str = "body_with_feet"
    default_detection_frequency: int = 4
    default_tracking_mode: str = "physiotrack"
    default_device: str = "auto"
    default_backend: str = "auto"
    
    # Security settings
    enable_auth: bool = False
    api_key: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    @validator("port")
    def port_must_be_valid(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    def get_upload_path(self):
        """Get upload directory path and create if doesn't exist"""
        import os
        os.makedirs(self.upload_dir, exist_ok=True)
        return self.upload_dir
    
    def get_results_path(self):
        """Get results directory path and create if doesn't exist"""
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        return self.results_dir
    
    def get_temp_path(self):
        """Get temp directory path and create if doesn't exist"""
        import os
        os.makedirs(self.temp_dir, exist_ok=True)
        return self.temp_dir

# Create a global settings instance
settings = Settings()