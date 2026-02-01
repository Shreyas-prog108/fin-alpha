from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """Agent configuration with security validation"""
    
    GROQ_API_KEY: str = Field(...)
    NEWSAPI_KEY: Optional[str] = Field(None)
    NEWSAPI_URL: str = Field("https://newsapi.org/v2")
    BACKEND_URL: Optional[str] = Field("http://localhost:8000")   
    API_KEY: Optional[str] = Field(None) 
    GROQ_MODEL: str = Field("gpt-oss-20b")
    LLM_TEMPERATURE: float = Field(1.0, ge=0.0, le=2.0)
    MAX_TOKENS: int = Field(2048, ge=1, le=100000)
    MAX_ITERATIONS: int = Field(5, ge=1, le=100)
    TOOL_TIMEOUT: int = Field(60, ge=5, le=600)
    MAX_LLM_CALLS_PER_QUERY: int = Field(10, ge=1, le=50)

    LOG_LEVEL: str = Field("INFO")
    ENABLE_DEBUG: bool = Field(False)
    
    class Config:
        env_file = ".env.local"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    @field_validator('BACKEND_URL')
    @classmethod
    def validate_backend_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("BACKEND_URL must start with http:// or https://")
        return v
    
    @field_validator('GROQ_API_KEY')
    @classmethod
    def validate_groq_key(cls, v):
        if not v or len(v) == 0:
            raise ValueError("GROQ_API_KEY must be set")
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()


try:
    config = AgentConfig()
    logger.info("Agent configuration loaded successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise