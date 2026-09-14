"""Perception boundary. Owner: ML.

Contract, frozen:  extract(event: dict) -> dict[str, Any]
Returns a partial slot dict. Returns {} on any failure. NEVER raises, never blocks
the agent loop -- the agent has already spoken by the time this runs.

Today this delegates to the deterministic keyword matcher in agent.py so the
harness stays reproducible. Replace the body, keep the signature.
"""

from agent import _extract as _keyword_extract


def extract(event: dict) -> dict:
    try:
        text = event.get("text") or event.get("caption") or ""
        return _keyword_extract(text)
    except Exception:
        return {}          # degrade to no-new-slots; never take the loop down


# --- to implement -----------------------------------------------------------
# transcribe(wav_bytes) -> str          faster-whisper, CPU
# caption(png_bytes)    -> str          small VLM
# extract via LLM       -> dict         same return type as above
