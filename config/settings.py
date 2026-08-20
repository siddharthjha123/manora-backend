"""
MANORA Backend - Configuration Settings
Defines environment variables, defaults, and runtime configuration settings.
"""

import os
from functools import lru_cache
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
        
        # Database (PostgreSQL / Supabase)
        DATABASE_URL: Optional[str] = None
        
        # LLM / OpenRouter
        OPENROUTER_API_KEY: Optional[str] = None
        OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
        MODEL_NAME: str = "openai/gpt-4o-mini"
        
        # ML Emotion Classifier
        EMOTION_MODEL_NAME: str = "distilbert-base-uncased-emotion"
        
        # Qdrant Vector DB
        QDRANT_URL: str = "http://localhost:6333"
        QDRANT_API_KEY: Optional[str] = None
        QDRANT_ENABLED: bool = False
        
        # Neo4j Graph DB
        NEO4J_URI: str = "bolt://localhost:7687"
        NEO4J_USERNAME: str = "neo4j"
        NEO4J_PASSWORD: str = "password"
        NEO4J_ENABLED: bool = False
        
        # Buddy State Configuration
        BUDDY_DECAY_RATE: float = 0.05
        
        # Application metadata
        APP_NAME: str = "MANORA Backend"
        APP_VERSION: str = "1.0.0"
        ENVIRONMENT: str = "development"
        LOG_LEVEL: str = "INFO"

        # Data Agent
        DATA_AGENT_API_KEY: Optional[str] = None
        DATA_AGENT_BASE_URL: str = "https://ai.tcetcercd.in/v1"
        DATA_AGENT_MODEL_NAME: str = "qwen3.6"
        DATA_AGENT_ENABLE_THINKING: bool = True
        DATA_AGENT_REASONING_EFFORT: str = "medium"

        # Observability - Sentry
        SENTRY_DSN: Optional[str] = None
        SENTRY_TRACES_SAMPLE_RATE: float = 1.0

        # Observability - Langfuse
        LANGFUSE_PUBLIC_KEY: Optional[str] = None
        LANGFUSE_SECRET_KEY: Optional[str] = None
        LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"

        # Observability - Prometheus
        PROMETHEUS_ENABLED: bool = True

except ImportError:
    # Fallback if pydantic-settings is not installed
    from pydantic import BaseModel

    class Settings(BaseModel):  # type: ignore
        DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
        OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        MODEL_NAME: str = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
        EMOTION_MODEL_NAME: str = os.getenv("EMOTION_MODEL_NAME", "distilbert-base-uncased-emotion")
        QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
        QDRANT_ENABLED: bool = os.getenv("QDRANT_ENABLED", "false").lower() in ("true", "1", "yes")
        NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
        NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
        NEO4J_ENABLED: bool = os.getenv("NEO4J_ENABLED", "false").lower() in ("true", "1", "yes")
        BUDDY_DECAY_RATE: float = float(os.getenv("BUDDY_DECAY_RATE", "0.05"))
        APP_NAME: str = "MANORA Backend"
        APP_VERSION: str = "1.0.0"
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        DATA_AGENT_API_KEY: Optional[str] = os.getenv("DATA_AGENT_API_KEY")
        DATA_AGENT_BASE_URL: str = os.getenv("DATA_AGENT_BASE_URL", "https://ai.tcetcercd.in/v1")
        DATA_AGENT_MODEL_NAME: str = os.getenv("DATA_AGENT_MODEL_NAME", "qwen3.6")
        DATA_AGENT_ENABLE_THINKING: bool = os.getenv("DATA_AGENT_ENABLE_THINKING", "true").lower() in ("true", "1", "yes")
        DATA_AGENT_REASONING_EFFORT: str = os.getenv("DATA_AGENT_REASONING_EFFORT", "medium")
        SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
        SENTRY_TRACES_SAMPLE_RATE: float = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))
        LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
        LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
        LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() in ("true", "1", "yes")


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()
