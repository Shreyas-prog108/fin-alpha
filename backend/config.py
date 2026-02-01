from fastapi import FastAPI
import os
from dotenv import load_dotenv
import logging

load_dotenv('.env.local', override=True)

logger = logging.getLogger(__name__)

#=========================
#CONFIG - SECURITY VALIDATED
#=========================
GROQ_MODEL=os.getenv("GROQ_MODEL", "gpt-oss-20b")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set - Groq features will be unavailable")

GROQ_API_URL="https://api.groq.com/openai/v1/chat/completions"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000").split(",")
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"

