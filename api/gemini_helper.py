from fastapi import HTTPException
import httpx
from config import GEMINI_API_KEY, GEMINI_API_URL

#=========================
#GEMINI HELPER
#=========================
async def query_gemini(prompt:str)->str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500,detail="GEMINI_API_KEY is not set")
    async with httpx.AsyncClient(timeout=60) as client:
        response=await client.post(
            GEMINI_API_URL,
            params={"key":GEMINI_API_KEY},
            json={"contents":[{"parts":[{"text":prompt}]}]},
        )
        if response.status_code!=200:
            raise HTTPException(status_code=500,detail=f"Gemini API error: {response.text}")
        body=response.json()
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise HTTPException(status_code=500,detail=f"Unexpected Gemini response:{body}")

