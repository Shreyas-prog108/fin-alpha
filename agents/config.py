from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

class AgentConfig(BaseSettings):
    GEMINI_API_KEY:str
    NEWSAPI_KEY:Optional[str]=None
    LLM_MODEL:str=os.getenv("GEMINI_MODEL","gemini-3-flash-preview")
    LLM_TEMPERATURE:float=1.0
    LLM_MAX_TOKENS:int=2048
    BACKEND_URL:str=os.getenv("BACKEND_URL","http://localhost:8000")
    MAX_ITERATIONS:int=5
    TOOL_TIMEOUTint=60
    ENABLE_CACHING:bool=True
    LOG_LEVEL:str="INFO"
    ENABLE_DEBUG:bool=False 
    class Config:
        env_file=".env.local"
        env_file_encoding="utf-8"
config=AgentConfig()