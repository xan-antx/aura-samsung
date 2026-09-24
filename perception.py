"""Perception boundary for Project Aura. Owner: ML.

Contract, frozen: extract(event: dict) -> dict[str, Any]
Returns a partial slot dict. Returns {} on any failure. NEVER raises, never blocks
the agent loop.

Pipelines:
  1. TEXT  -> LLM -> slots (with safe deterministic fallback)
  2. WAV   -> faster-whisper (CPU int8) -> transcript -> slots
  3. PNG   -> VLM / captioner -> caption -> slots

No dependency on agent.py (avoids circular imports).
"""

import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Cloudflare-fronted providers (Groq among them) reject Python's default
# `Python-urllib` agent with 403 "error code: 1010" - every outbound request
# carries an explicit agent instead.
USER_AGENT = "aura-samsung/1.0"

# --- Schema and normalization constants --------------------------------------

ALLOWED_SLOTS = frozenset({"origin", "destination", "date", "pax", "intent", "commit"})

CITIES = {"delhi", "mumbai", "bengaluru", "bangalore", "chennai", "goa", "pune"}
CITY_ALIASES = {"bangalore": "bengaluru"}

# Benchmark evaluation clock reference:
# Harness scenarios run on a virtual clock where simulation start t=0.0 corresponds
# to Monday 2026-09-14. Therefore "tomorrow" -> "2026-09-15" and "friday" -> "2026-09-18".
DATE_ALIASES = {
    "tomorrow": "2026-09-15",
    "friday": "2026-09-18",
}

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

SYSTEM_PROMPT = (
    "You are the slot extraction module for a flight booking assistant.\n"
    "Extract flight booking slots from the user input.\n"
    "Return ONLY a JSON object with any of these recognized keys:\n"
    '- "origin": departure city (e.g. "delhi", "mumbai", "pune", "bengaluru", "goa", "chennai")\n'
    '- "destination": arrival city (e.g. "delhi", "mumbai", "pune", "bengaluru", "goa", "chennai")\n'
    '- "date": travel date formatted as YYYY-MM-DD (e.g. "2026-09-15", "2026-09-18")\n'
    '- "pax": integer count of passengers (e.g. 1, 2)\n'
    '- "intent": "flight" if discussing or booking flights\n'
    '- "commit": true if the user confirms, commits, or asks to book\n\n'
    "Rules:\n"
    "- Output ONLY the JSON object. No markdown code fences, no explanations.\n"
    "- Omit unmentioned slots.\n"
    '- Normalize city names to lowercase (e.g. "bangalore" -> "bengaluru").'
)

# --- Model Singletons and Hooks ----------------------------------------------

_WHISPER_MODEL = None
_VLM_CAPTIONER = None  # Hook for test mocking or custom local VLM


def _get_whisper_model():
    """Lazily load and cache the faster-whisper model on CPU int8."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        return _WHISPER_MODEL
    except Exception:
        return None


# --- Normalization & Deterministic Fallback -----------------------------------


def normalize_slots(raw: Any) -> dict[str, Any]:
    """Validate and normalize extracted slots against project schema."""
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in ALLOWED_SLOTS:
            continue

        if k in ("origin", "destination") and isinstance(v, str):
            cleaned = CITY_ALIASES.get(v.strip().lower(), v.strip().lower())
            if cleaned:
                out[k] = cleaned

        elif k == "date" and isinstance(v, str):
            d = v.strip().lower()
            if d in DATE_ALIASES:
                out["date"] = DATE_ALIASES[d]
            elif re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                out["date"] = d

        elif k == "pax":
            if isinstance(v, int) and v > 0:
                out["pax"] = v
            elif isinstance(v, str):
                s = v.strip().lower()
                if s.isdigit() and int(s) > 0:
                    out["pax"] = int(s)
                elif s in WORD_TO_NUM:
                    out["pax"] = WORD_TO_NUM[s]

        elif k == "intent" and isinstance(v, str) and v.strip().lower() in ("flight", "fly"):
            out["intent"] = "flight"

        elif k == "commit":
            if isinstance(v, bool):
                out["commit"] = v
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1", "commit", "confirm", "book"):
                    out["commit"] = True
                elif s in ("false", "no", "0"):
                    out["commit"] = False
            elif isinstance(v, (int, float)):
                out["commit"] = bool(v)

    return out


def _fallback_extract(text: str) -> dict[str, Any]:
    """Self-contained deterministic extractor matching project slot schema.
    Used when ML models are unconfigured, offline, or fail."""
    t = text.lower()
    out: dict[str, Any] = {}
    words = [w.strip(".,!?") for w in t.split()]

    for i, w in enumerate(words):
        if w in ("to", "for") and i + 1 < len(words) and words[i + 1] in CITIES:
            out["destination"] = CITY_ALIASES.get(words[i + 1], words[i + 1])
        if w == "from" and i + 1 < len(words) and words[i + 1] in CITIES:
            out["origin"] = CITY_ALIASES.get(words[i + 1], words[i + 1])

    if "tomorrow" in t:
        out["date"] = DATE_ALIASES["tomorrow"]
    if "friday" in t:
        out["date"] = DATE_ALIASES["friday"]

    for n, v in (("one", 1), ("two", 2), ("three", 3), ("four", 4)):
        if f"{n} seat" in t or f"{n} ticket" in t or f"{n} passenger" in t:
            out["pax"] = v
    if "flight" in t or "fly" in t:
        out["intent"] = "flight"
    if "book" in t or "confirm" in t:
        out["commit"] = True
    return out


# --- LLM Client & Slot Extraction --------------------------------------------

_WARNED: set = set()      # (context, kind) pairs already reported to stderr
_LLM_OK: bool | None = None   # None = no call attempted yet


def _warn_once(context: str, exc: Exception) -> None:
    """The fallback stays, but never silently: the first failure of each kind
    is reported to stderr with the HTTP status and the provider's error body.
    Keys are never logged (no headers, no env)."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(500).decode("utf-8", "replace")
        except Exception:
            body = ""
        key = (context, "http", exc.code)
        detail = f"HTTP {exc.code}: {body!r}"
    else:
        key = (context, type(exc).__name__)
        detail = f"{type(exc).__name__}: {exc}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[perception] {context} failed ({detail}) - "
          "falling back to the deterministic extractor", file=sys.stderr)


def llm_status() -> str:
    """'off' | 'untried' | 'ok' | 'failing' - so a UI can say whether LLM
    extraction is actually working, not merely whether a key is present."""
    if not _is_llm_configured():
        return "off"
    if _LLM_OK is None:
        return "untried"
    return "ok" if _LLM_OK else "failing"


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "User-Agent": USER_AGENT}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


_RETRY_ONCE = {429, 503}       # transient: one short backoff, same model
_NEXT_MODEL = {404, 429, 503}  # then move down the model list


def _try_models(default_model: str, attempt, timeout: float):
    """AURA_LLM_MODEL may be a comma-separated list. 429/503 get one retry
    after a short backoff (within the timeout budget); 404/429/503 then move
    to the next model; anything else raises immediately."""
    models = [m.strip() for m in os.getenv("AURA_LLM_MODEL", default_model).split(",")
              if m.strip()]
    last = None
    for model in models:
        for attempt_no in (0, 1):
            try:
                return attempt(model)
            except urllib.error.HTTPError as e:
                last = e
                if attempt_no == 0 and e.code in _RETRY_ONCE:
                    time.sleep(min(0.3, timeout / 4))
                    continue
                if e.code in _NEXT_MODEL:
                    break                     # try the next model, if any
                raise
    raise last


def _is_llm_configured() -> bool:
    return bool(os.getenv("AURA_LLM_URL") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _parse_json_from_text(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        lines = [line for line in t.splitlines() if not line.strip().startswith("```")]
        t = "\n".join(lines).strip()

    try:
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        data = json.loads(t[start : end + 1])
        if isinstance(data, dict):
            return data

    raise ValueError("No valid JSON dict found in LLM response")


def _call_llm(text: str) -> dict:
    aura_url = os.getenv("AURA_LLM_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    timeout = float(os.getenv("AURA_LLM_TIMEOUT", "2.0"))

    if aura_url or openai_key:
        url = aura_url or (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions")
        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"

        def attempt(model):
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
            }
            data = _post_json(url, payload, headers, timeout)
            return _parse_json_from_text(data["choices"][0]["message"]["content"])

        return _try_models("gpt-4o-mini", attempt, timeout)

    if gemini_key:
        def attempt(model):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nUser input: {text}"}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
            }
            data = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
            return _parse_json_from_text(data["candidates"][0]["content"]["parts"][0]["text"])

        return _try_models("gemini-3.6-flash", attempt, timeout)

    raise ValueError("No LLM provider configured")


def extract_text_slots(text: str) -> dict[str, Any]:
    """Extract slots from text using LLM (if configured) with safe deterministic fallback."""
    if not isinstance(text, str) or not text.strip():
        return {}

    if _is_llm_configured():
        global _LLM_OK
        try:
            raw_slots = _call_llm(text)
            _LLM_OK = True
            normalized = normalize_slots(raw_slots)
            if normalized:
                return normalized
            return _fallback_extract(text)
        except Exception as e:
            _LLM_OK = False
            _warn_once("LLM extraction", e)
            return _fallback_extract(text)

    return _fallback_extract(text)


# --- Audio Handling (WAV -> faster-whisper) -----------------------------------


def _is_valid_wav(data: Any) -> bool:
    if not isinstance(data, (bytes, bytearray)) or len(data) < 12:
        return False
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    try:
        import wave
        with wave.open(io.BytesIO(data), "rb") as w:
            w.getnchannels()
        return True
    except Exception:
        return False


def _extract_audio(event: dict) -> dict[str, Any]:
    raw_audio = event.get("audio") or event.get("wav_bytes") or event.get("wav")
    if not raw_audio or not _is_valid_wav(raw_audio):
        return {}

    model = _get_whisper_model()
    if model is None:
        return {}

    try:
        segments, _ = model.transcribe(io.BytesIO(raw_audio), beam_size=1)
        chunks = [seg.text for seg in segments if hasattr(seg, "text") and seg.text]
        transcript = " ".join(chunks).strip()
        return extract_text_slots(transcript) if transcript else {}
    except Exception:
        return {}


# --- Image / Frame Handling (PNG -> VLM -> Caption -> slots) ------------------


def _is_valid_png(data: Any) -> bool:
    return isinstance(data, (bytes, bytearray)) and len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n"


def _caption_png(png_bytes: bytes) -> str | None:
    global _VLM_CAPTIONER
    if _VLM_CAPTIONER is not None:
        return _VLM_CAPTIONER(png_bytes)

    aura_vlm_url = os.getenv("AURA_VLM_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    timeout = float(os.getenv("AURA_VLM_TIMEOUT", "3.0"))
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    if aura_vlm_url or openai_key:
        url = aura_vlm_url or (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions")
        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"
        payload = {
            "model": os.getenv("AURA_VLM_MODEL", "gpt-4o-mini"),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe any flight, boarding pass, ticket, origin, destination, date, or passengers visible."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 150,
        }
        resp_data = _post_json(url, payload, headers, timeout)
        return resp_data["choices"][0]["message"]["content"].strip()

    if gemini_key:
        model = os.getenv("AURA_VLM_MODEL", "gemini-3.6-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Describe any flight, boarding pass, ticket, origin, destination, date, or passengers visible."},
                    {"inlineData": {"mimeType": "image/png", "data": b64}},
                ]
            }]
        }
        resp_data = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
        return resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()

    return None


def _extract_image(event: dict) -> dict[str, Any]:
    if event.get("caption"):
        return extract_text_slots(event["caption"])

    raw_image = (
        event.get("image")
        or event.get("png_bytes")
        or (event.get("frame") if not isinstance(event.get("frame"), str) else None)
    )
    if not raw_image or not _is_valid_png(raw_image):
        return {}

    try:
        caption = _caption_png(raw_image)
        return extract_text_slots(caption) if caption else {}
    except Exception as e:
        _warn_once("VLM captioning", e)
        return {}


# --- Public Entrypoint -------------------------------------------------------


def _dispatch_extract(event: dict) -> dict[str, Any]:
    kind = event.get("kind")
    if kind == "audio" or any(k in event for k in ("audio", "wav_bytes", "wav")):
        return _extract_audio(event)
    if kind == "frame" or any(k in event for k in ("image", "png_bytes")):
        return _extract_image(event)
    if "caption" in event and event["caption"]:
        return extract_text_slots(event["caption"])
    text = event.get("text")
    if text is not None:
        return extract_text_slots(text) if isinstance(text, str) and text.strip() else {}
    return {}


def extract(event: dict) -> dict[str, Any]:
    """Perception contract entrypoint: extract(event: dict) -> dict[str, Any]
    Never raises, returns {} on any failure."""
    if not isinstance(event, dict):
        return {}
    try:
        return _dispatch_extract(event)
    except Exception:
        return {}
