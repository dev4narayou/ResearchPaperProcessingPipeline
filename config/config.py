from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # API Keys and Authentication
    UNPAYWALL_EMAIL: str
    OPENAI_API_KEY: str

    # Database Configuration
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5439/example_db"

    # Rate Limiting
    MAX_CONCURRENT_DOWNLOADS: int = 10
    MAX_CONCURRENT_LLM_CALLS: int = 5

    # Queue Sizes
    QUEUE_MAX_SIZE: int = 200

    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 4
    RETRY_MAX_WAIT: int = 10

    # LLM Configuration
    LLM_MODEL: str = "gpt-4-turbo-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# create a global settings instance
settings = get_settings()