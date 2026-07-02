from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os, httpx, base64
from google import genai
from google.genai import types

app = FastAPI()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
    async with httpx.AsyncClient() as http:
        r = await http.get(url)
        img_b64 = base64.b64encode(r.content).decode()

    response = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_b64)),
            types.Part(text="Analyze this field/crop image. Describe plant health, any visible diseases or stress signs, and give a brief recommendation. Keep it under 100 words.")
        ]
    )
    latest["url"] = url
    latest["analysis"] = response.text
    return {"status": "ok", "analysis": response.text}

@app.get("/latest")
async def get_latest():
    return latest
