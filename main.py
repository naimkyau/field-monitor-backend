from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os, httpx, base64, re
from google import genai
from google.genai import types
 
app = FastAPI()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
latest = {
    "url":      None,
    "status":   "Waiting...",
    "analysis": "Waiting for first image...",
    "disease":  "N/A",
    "desc":     "No data yet."
}
 
class ImagePayload(BaseModel):
    image_url: str
 
@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())
 
@app.post("/upload")
async def upload(payload: ImagePayload):
    url = payload.image_url
    async with httpx.AsyncClient() as http:
        r = await http.get(url)
        img_b64 = base64.b64encode(r.content).decode()
 
    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_b64)),
            types.Part(text="""Analyze this crop/field image for plant disease detection.
 
Reply in EXACTLY this format (no extra text, no markdown):
STATUS: <Healthy | Diseased | Stressed | Unknown>
DISEASE: <disease name or 'None' if healthy>
DESC: <one sentence description, max 15 words>
DETAIL: <full analysis under 80 words>
 
Example:
STATUS: Diseased
DISEASE: Early Blight
DESC: Fungal infection causing dark spots on lower leaves.
DETAIL: Plant shows classic early blight symptoms with dark brown concentric ring lesions on lower leaves. Caused by Alternaria solani. Remove affected leaves, apply copper-based fungicide, avoid overhead watering. Early intervention can prevent spread to upper foliage.""")
        ]
    )
 
    text = response.text.strip()
 
    # Parse structured fields
    def extract(tag):
        m = re.search(rf"^{tag}:\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else "Unknown"
 
    status  = extract("STATUS")
    disease = extract("DISEASE")
    desc    = extract("DESC")
    detail  = extract("DETAIL")
 
    latest["url"]      = url
    latest["status"]   = status
    latest["disease"]  = disease
    latest["desc"]     = desc
    latest["analysis"] = detail
 
    return {
        "status":  "ok",
        "plant_status": status,
        "disease": disease,
        "desc":    desc,
        "analysis": detail
    }
 
@app.get("/latest")
async def get_latest():
    return latest
 
# ── OLED endpoint ──────────────────────────────────────────────────────────────
# ESP32 polls this every 60s to get compact OLED display data
@app.get("/oled")
async def oled_data():
    """Returns minimal JSON optimised for ESP32 OLED display parsing."""
    return {
        "disease": latest["disease"][:20],   # max 20 chars — fits OLED line
        "desc":    latest["desc"][:80]        # up to 80 chars for word-wrap
    }
