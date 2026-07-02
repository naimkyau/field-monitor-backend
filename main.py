from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.genai as genai
import os, httpx, base64

app = FastAPI()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

latest = {"url": None, "analysis": "Waiting for first image..."}

class ImagePayload(BaseModel):
    image_url: str

@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.post("/upload")
async def upload(payload: ImagePayload):
    url = payload.image_url

    # ImageKit থেকে image download করে Gemini-তে দেওয়া
    async with httpx.AsyncClient() as http:
        r = await http.get(url)
        img_b64 = base64.b64encode(r.content).decode()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    {"text": "Analyze this field/crop image. Describe plant health, any visible diseases or stress signs, and give a brief recommendation. Keep it under 100 words."}
                ]
            }
        ]
    )

    latest["url"] = url
    latest["analysis"] = response.text
    return {"status": "ok", "analysis": response.text}

@app.get("/latest")
async def get_latest():
    return latest
