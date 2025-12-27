from fastapi import FastAPI
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

#=========================
#CONFIG - SECURITY VALIDATED
#=========================
GEMINI_API_VERSION=os.getenv("GEMINI_API_VERSION", "v1beta")
GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set - Gemini features will be unavailable")

GEMINI_API_URL=(
    f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/"
    f"{GEMINI_MODEL}:generateContent"
)
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"
API_KEY = os.getenv("API_KEY")
if not API_KEY and not DEBUG_MODE:
    logger.error("API_KEY not set in production mode - API will be insecure!")
    raise ValueError("API_KEY environment variable must be set in production")
