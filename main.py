from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
import os, httpx, base64, hashlib, hmac, time
from google import genai
from google.genai import types

app = FastAPI()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── Secrets (set these in Render environment variables) ──────────────────────
IMAGEKIT_PRIVATE_KEY = os.environ["IMAGEKIT_PRIVATE_KEY"]   # e.g. private_xxx
DEVICE_SECRET        = os.environ["DEVICE_SECRET"]          # shared secret with ESP32

IMAGEKIT_UPLOAD_URL  = "https://upload.imagekit.io/api/v1/files/upload"

latest = {"url": None, "analysis": "Waiting for first image...", "timestamp": None}

# ── Helpers ──────────────────────────────────────────────────────────────────

def verify_hmac(body: bytes, timestamp: str, received_sig: str) -> bool:
    """
    ESP32 sends:  HMAC-SHA256(secret, timestamp + body_bytes)
    We recompute the same and compare.
    Reject requests with a timestamp older than 60 seconds.
    """
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False
    if abs(time.time() - ts) > 60:
        return False                         # replay protection
    message = timestamp.encode() + body
    expected = hmac.new(
        DEVICE_SECRET.encode(), message, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, received_sig or "")


async def upload_to_imagekit(image_bytes: bytes, filename: str) -> str:
    """Upload raw JPEG bytes to ImageKit and return the public URL."""
    b64 = base64.b64encode(image_bytes).decode()
    auth = base64.b64encode((IMAGEKIT_PRIVATE_KEY + ":").encode()).decode()

    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(
            IMAGEKIT_UPLOAD_URL,
            headers={"Authorization": f"Basic {auth}"},
            data={"fileName": filename},
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
    r.raise_for_status()
    return r.json()["url"]


async def analyze_with_gemini(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    response = gemini.models.generate_content(
        model="gemini-2.0-flash",          # ← correct model string
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
            types.Part(text=(
                "Analyze this field/crop image. Describe plant health, "
                "any visible diseases or stress signs, and give a brief "
                "recommendation. Keep it under 100 words."
            )),
        ],
    )
    return response.text


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.post("/upload")
async def upload(
    request: Request,
    image: UploadFile = File(...),
    x_timestamp:  str = Header(None),
    x_signature:  str = Header(None),
):
    """
    ESP32 POSTs multipart/form-data with:
      - file field  : 'image'  (raw JPEG bytes)
      - header      : X-Timestamp  (Unix seconds as string)
      - header      : X-Signature  (HMAC-SHA256 hex)
    """
    body = await image.read()

    # ── Auth check ─────────────────────────────────────────────────────────
    if not verify_hmac(body, x_timestamp, x_signature):
        raise HTTPException(status_code=401, detail="Invalid or expired signature")

    filename = f"cap_{x_timestamp}.jpg"

    # ── Upload to ImageKit ──────────────────────────────────────────────────
    try:
        image_url = await upload_to_imagekit(body, filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ImageKit upload failed: {e}")

    # ── Gemini analysis ─────────────────────────────────────────────────────
    try:
        analysis = await analyze_with_gemini(body)
    except Exception as e:
        analysis = f"AI analysis failed: {e}"

    from datetime import datetime, timezone
    latest["url"]       = image_url
    latest["analysis"]  = analysis
    latest["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {"status": "ok", "url": image_url, "analysis": analysis}


@app.get("/latest")
async def get_latest():
    return latest
