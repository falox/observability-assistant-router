"""Pydantic settings configuration for the router."""

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 9010
    log_level: LogLevel = LogLevel.INFO

    # Configuration path
    config_path: str = "/config/agents.yaml"

    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retry settings
    retry_attempts: int = 2
    retry_backoff_ms: int = 500

    # Session settings
    session_enabled: bool = True
    session_timeout_min: int = 30
    session_drift_threshold: float = 0.5

    # Observability settings
    audit_enabled: bool = True
    audit_log_level: str = "INFO"
    stream_buffer_enabled: bool = True
    stream_buffer_max_size: int = 1_000_000  # 1MB max per message

    # Hot reload settings
    hot_reload_enabled: bool = True
    hot_reload_debounce_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
