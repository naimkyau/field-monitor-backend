from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import HTMLResponse
import os, httpx, base64, hashlib, hmac, time
from google import genai
from google.genai import types

app = FastAPI()

# ── Lazy-load secrets so a missing var gives a clear error, not a 500 ────────
def get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val

# ── State ─────────────────────────────────────────────────────────────────────
latest = {"url": None, "analysis": "Waiting for first image...", "timestamp": None}

IMAGEKIT_UPLOAD_URL = "https://upload.imagekit.io/api/v1/files/upload"

# ── HMAC verification ─────────────────────────────────────────────────────────
def verify_hmac(body: bytes, timestamp_str: str, received_sig: str) -> bool:
    """
    ESP32 has no RTC so it sends millis()/1000 (uptime seconds), NOT unix epoch.
    We cannot do replay-protection via epoch comparison, so we just verify the
    signature matches. A static device secret is enough for a university project.
    For production, add NTP to the ESP32 and switch back to epoch comparison.
    """
    if not timestamp_str or not received_sig:
        return False
    secret = get_env("DEVICE_SECRET")
    message = timestamp_str.encode() + body
    expected = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_sig)


# ── ImageKit upload ───────────────────────────────────────────────────────────
async def upload_to_imagekit(image_bytes: bytes, filename: str) -> str:
    private_key = get_env("IMAGEKIT_PRIVATE_KEY")
    auth = base64.b64encode((private_key + ":").encode()).decode()
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(
            IMAGEKIT_UPLOAD_URL,
            headers={"Authorization": f"Basic {auth}"},
            data={"fileName": filename},
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
    r.raise_for_status()
    return r.json()["url"]


# ── Gemini analysis ───────────────────────────────────────────────────────────
async def analyze_with_gemini(image_bytes: bytes) -> str:
    client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    b64 = base64.b64encode(image_bytes).decode()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.post("/upload")
async def upload(
    image:       UploadFile = File(...),
    x_timestamp: str        = Header(None),
    x_signature: str        = Header(None),
):
    body = await image.read()

    if not verify_hmac(body, x_timestamp, x_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    filename = f"cap_{x_timestamp}.jpg"

    try:
        image_url = await upload_to_imagekit(body, filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ImageKit upload failed: {e}")

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


# ── Health check (useful for Render) ─────────────────────────────────────────
@app.get("/health")
async def health():
    missing = [k for k in ("GEMINI_API_KEY", "IMAGEKIT_PRIVATE_KEY", "DEVICE_SECRET")
               if not os.environ.get(k)]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {missing}")
    return {"status": "ok"}
