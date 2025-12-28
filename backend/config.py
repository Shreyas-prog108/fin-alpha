from fastapi import FastAPI
import os
from dotenv import load_dotenv
import logging

load_dotenv('.env.local', override=True)

logger = logging.getLogger(__name__)

#=========================
#CONFIG - SECURITY VALIDATED
#=========================
GEMINI_API_VERSION=os.getenv("GEMINI_API_VERSION", "v1beta")
GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set - Gemini features will be unavailable")

GEMINI_API_URL=(
    f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/"
    f"{GEMINI_MODEL}:generateContent"
)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000").split(",")
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
