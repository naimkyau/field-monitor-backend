import os
import io
import httpx
import google.generativeai as genai
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Gemini setup — key comes from Render environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

PROMPT = """
You are an agricultural expert. Analyze this crop/plant image carefully.

Respond in this exact format:
Status: [Healthy / Diseased / Stressed / Unclear]
Issue: [disease or problem name, or "None"]
Severity: [Low / Medium / High / N/A]
Action: [short practical advice for the farmer]

Be concise. Max 2 sentences for Action.
"""

# In-memory store — holds the latest capture
latest = {
    "url": None,
    "analysis": None,
    "timestamp": None,
    "status": None
}


class ImagePayload(BaseModel):
    url: str


@app.post("/upload")
async def receive_image(payload: ImagePayload):
    """ESP32 calls this after uploading to ImageKit."""
    try:
        # Download image from ImageKit
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(payload.url)
            r.raise_for_status()
            image_bytes = r.content

        # Open with Pillow and send to Gemini Vision
        image = Image.open(io.BytesIO(image_bytes))
        result = model.generate_content([PROMPT, image])
        analysis = result.text.strip()

        # Parse status line for color coding on frontend
        status = "Unknown"
        for line in analysis.splitlines():
            if line.lower().startswith("status:"):
                status = line.split(":", 1)[1].strip()
                break

        # Save latest
        latest["url"] = payload.url
        latest["analysis"] = analysis
        latest["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        latest["status"] = status

        return {"ok": True, "analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest")
async def get_latest():
    """Webpage polls this endpoint every 30 seconds."""
    return JSONResponse(latest)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()
