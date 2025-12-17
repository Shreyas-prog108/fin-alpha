from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()
#=========================
#CONFIG
#=========================
GEMINI_API_VERSION=os.getenv("GEMINI_API_VERSION","v1beta")
GEMINI_MODEL=os.getenv("GEMINI_MODEL","gemini-3-flash-preview")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
GEMINI_API_URL=(
    f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/"
    f"{GEMINI_MODEL}:generateContent"
)
