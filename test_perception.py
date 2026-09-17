"""Unit tests and self-checks for perception.py.

Covers all required test cases across Text, Audio, Image, and General boundaries.
Runs purely on Python standard library with no external dependencies required.
"""

import io
import os
import subprocess
import sys
import unittest
import wave
from unittest.mock import MagicMock, patch

import perception
from perception import (
    extract,
    extract_text_slots,
    normalize_slots,
    _parse_json_from_text,
)

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def make_dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


class MockWhisperSegment:
    def __init__(self, text: str):
        self.text = text


class TestPerception(unittest.TestCase):
    def setUp(self):
        perception._WHISPER_MODEL = None
        perception._VLM_CAPTIONER = None
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        perception._WHISPER_MODEL = None
        perception._VLM_CAPTIONER = None

    # --- CIRCULAR IMPORT CHECK ------------------------------------------------

    def test_00_perception_imports_independently_without_agent(self):
        """0. Perception must import in isolation without loading agent.py."""
        code = (
            "import sys; "
            "import perception; "
            "assert 'agent' not in sys.modules, "
            "f'agent was unexpectedly imported by perception: {sys.modules.get(\"agent\")}'"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Import isolation check failed: {res.stderr}")

    # --- TEXT -----------------------------------------------------------------

    def test_01_valid_text_extraction_produces_dictionary(self):
        """1. Valid text extraction produces a dictionary."""
        ev = {"kind": "chunk", "text": "I need a flight from Delhi to Mumbai tomorrow"}
        res = extract(ev)
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("origin"), "delhi")
        self.assertEqual(res.get("destination"), "mumbai")
        self.assertEqual(res.get("date"), "2026-09-15")

    def test_02_llm_structured_response_normalized_correctly(self):
        """2. LLM structured response is normalized correctly."""
        os.environ["AURA_LLM_URL"] = "http://mock-llm-server"
        mock_raw = {
            "origin": "DELHI",
            "destination": "Bangalore",
            "date": "tomorrow",
            "pax": "two",
            "intent": "fly",
            "commit": "true",
            "unrecognized_slot": "ignored",
        }
        with patch.object(perception, "_call_llm", return_value=mock_raw):
            res = extract({"kind": "chunk", "text": "book flight from Delhi to Bangalore tomorrow for two passengers"})
            self.assertEqual(res.get("origin"), "delhi")
            self.assertEqual(res.get("destination"), "bengaluru")
            self.assertEqual(res.get("date"), "2026-09-15")
            self.assertEqual(res.get("pax"), 2)
            self.assertEqual(res.get("intent"), "flight")
            self.assertIs(res.get("commit"), True)
            self.assertNotIn("unrecognized_slot", res)

    def test_03_llm_malformed_json_returns_empty_or_fallback(self):
        """3. LLM malformed JSON returns {} or safely uses fallback."""
        os.environ["AURA_LLM_URL"] = "http://mock-llm-server"
        with patch.object(perception, "_call_llm", side_effect=ValueError("malformed JSON")):
            # Fallback parses keywords if any exist
            res_with_kw = extract({"kind": "chunk", "text": "flight to Goa tomorrow"})
            self.assertEqual(res_with_kw.get("destination"), "goa")

            # Fallback on text without keywords safely returns {}
            res_no_kw = extract({"kind": "chunk", "text": "hello how are you"})
            self.assertEqual(res_no_kw, {})

        with self.assertRaises(ValueError):
            _parse_json_from_text("This is not valid json")

    def test_03b_llm_configured_but_returns_empty_output(self):
        """3b. LLM configured but returns empty/whitespace or empty dict falls back safely."""
        os.environ["AURA_LLM_URL"] = "http://mock-llm-server"
        with patch.object(perception, "_call_llm", return_value={}):
            # When LLM returns {}, falls back to keyword extractor
            res = extract({"kind": "chunk", "text": "flight from Delhi to Goa tomorrow"})
            self.assertEqual(res.get("origin"), "delhi")
            self.assertEqual(res.get("destination"), "goa")
            self.assertEqual(res.get("date"), "2026-09-15")

            # Text with no recognizable slots returns {}
            res_empty = extract({"kind": "chunk", "text": "just saying hi"})
            self.assertEqual(res_empty, {})

    def test_04_missing_api_configuration_does_not_crash(self):
        """4. Missing API configuration does not crash."""
        os.environ.pop("AURA_LLM_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        self.assertFalse(perception._is_llm_configured())
        res = extract({"kind": "chunk", "text": "flight from Pune to Chennai on Friday"})
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("origin"), "pune")
        self.assertEqual(res.get("destination"), "chennai")
        self.assertEqual(res.get("date"), "2026-09-18")

    def test_05_unknown_slots_are_rejected(self):
        """5. Unknown slots are rejected."""
        raw = {
            "origin": "delhi",
            "destination": "mumbai",
            "airline": "Air India",
            "meal": "vegetarian",
            "seat_class": "business",
            "pax": 1,
        }
        normalized = normalize_slots(raw)
        self.assertEqual(set(normalized.keys()), {"origin", "destination", "pax"})
        self.assertNotIn("airline", normalized)
        self.assertNotIn("meal", normalized)
        self.assertNotIn("seat_class", normalized)

    def test_06_pax_strings_are_normalized_to_integers(self):
        """6. pax strings are normalized to integers."""
        self.assertEqual(normalize_slots({"pax": "one"}), {"pax": 1})
        self.assertEqual(normalize_slots({"pax": "two"}), {"pax": 2})
        self.assertEqual(normalize_slots({"pax": "three"}), {"pax": 3})
        self.assertEqual(normalize_slots({"pax": "4"}), {"pax": 4})
        self.assertEqual(normalize_slots({"pax": 2}), {"pax": 2})
        self.assertEqual(normalize_slots({"pax": "unlimited"}), {})
        self.assertEqual(normalize_slots({"pax": -1}), {})

    def test_07_city_aliases_are_normalized(self):
        """7. city aliases are normalized."""
        norm1 = normalize_slots({"destination": "Bangalore", "origin": "BENGALURU"})
        self.assertEqual(norm1["destination"], "bengaluru")
        self.assertEqual(norm1["origin"], "bengaluru")

        norm2 = normalize_slots({"origin": "  mumbai  ", "destination": "Goa"})
        self.assertEqual(norm2["origin"], "mumbai")
        self.assertEqual(norm2["destination"], "goa")

    # --- AUDIO ----------------------------------------------------------------

    def test_08_missing_audio_returns_empty(self):
        """8. Missing audio returns {}."""
        self.assertEqual(extract({"kind": "audio"}), {})
        self.assertEqual(extract({"kind": "audio", "audio": None}), {})
        self.assertEqual(extract({"kind": "audio", "wav_bytes": b""}), {})

    def test_09_invalid_wav_returns_empty(self):
        """9. Invalid WAV returns {}."""
        self.assertEqual(extract({"kind": "audio", "audio": b"not a wav file header"}), {})
        self.assertEqual(extract({"kind": "audio", "audio": b"RIFFshort"}), {})

    def test_10_missing_faster_whisper_does_not_crash_import(self):
        """10. Missing faster-whisper does not crash import."""
        with patch.dict(sys.modules, {"faster_whisper": None}):
            res = perception._get_whisper_model()
            self.assertIsNone(res)
            self.assertEqual(extract({"kind": "audio", "audio": make_dummy_wav()}), {})

    def test_11_whisper_initialization_failure_returns_empty(self):
        """11. Whisper initialization failure returns {}."""
        with patch.object(perception, "_get_whisper_model", return_value=None):
            res = extract({"kind": "audio", "audio": make_dummy_wav()})
            self.assertEqual(res, {})

    def test_12_whisper_transcription_failure_returns_empty(self):
        """12. Whisper transcription failure returns {}."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CPU ASR failed")
        with patch.object(perception, "_get_whisper_model", return_value=mock_model):
            res = extract({"kind": "audio", "audio": make_dummy_wav()})
            self.assertEqual(res, {})

    def test_13_successful_transcription_passed_to_slot_extraction(self):
        """13. Successful transcription is passed to slot extraction."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [MockWhisperSegment("I need a flight from Delhi to Goa tomorrow for two passengers")],
            None,
        )
        with patch.object(perception, "_get_whisper_model", return_value=mock_model):
            res = extract({"kind": "audio", "audio": make_dummy_wav()})
            self.assertEqual(res.get("origin"), "delhi")
            self.assertEqual(res.get("destination"), "goa")
            self.assertEqual(res.get("date"), "2026-09-15")
            self.assertEqual(res.get("pax"), 2)

    # --- IMAGE ----------------------------------------------------------------

    def test_14_missing_image_returns_empty(self):
        """14. Missing image returns {}."""
        self.assertEqual(extract({"kind": "frame"}), {})
        self.assertEqual(extract({"kind": "frame", "image": None}), {})
        self.assertEqual(extract({"kind": "frame", "png_bytes": b""}), {})

    def test_15_invalid_png_returns_empty(self):
        """15. Invalid PNG returns {}."""
        self.assertEqual(extract({"kind": "frame", "image": b"corrupt image data"}), {})

    def test_16_vlm_unavailable_returns_empty(self):
        """16. VLM unavailable returns {}."""
        os.environ.pop("AURA_VLM_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        perception._VLM_CAPTIONER = None

        res = extract({"kind": "frame", "image": MINIMAL_PNG})
        self.assertEqual(res, {})

    def test_17_vlm_failure_returns_empty(self):
        """17. VLM failure returns {}."""
        def failing_captioner(_):
            raise ConnectionError("VLM host unreachable")

        perception._VLM_CAPTIONER = failing_captioner
        res = extract({"kind": "frame", "image": MINIMAL_PNG})
        self.assertEqual(res, {})

    def test_18_existing_caption_events_still_work(self):
        """18. Existing caption events still work without requiring VLM."""
        ev = {"kind": "frame", "caption": "boarding pass from Delhi to Pune"}
        res = extract(ev)
        self.assertEqual(res.get("origin"), "delhi")
        self.assertEqual(res.get("destination"), "pune")

    def test_19_successful_vlm_caption_passed_to_slot_extraction(self):
        """19. Successful VLM caption is passed to slot extraction."""
        perception._VLM_CAPTIONER = lambda _: "Boarding ticket for flight from Mumbai to Chennai on Friday"
        res = extract({"kind": "frame", "image": MINIMAL_PNG})
        self.assertEqual(res.get("origin"), "mumbai")
        self.assertEqual(res.get("destination"), "chennai")
        self.assertEqual(res.get("date"), "2026-09-18")

    # --- GENERAL --------------------------------------------------------------

    def test_20_extract_none_safely_returns_empty(self):
        """20. extract(None) safely returns {}."""
        self.assertEqual(extract(None), {})

    def test_21_malformed_events_safely_return_empty(self):
        """21. malformed events safely return {}."""
        self.assertEqual(extract("not a dict"), {})
        self.assertEqual(extract([1, 2, 3]), {})
        self.assertEqual(extract(42), {})
        self.assertEqual(extract({"random_field": 123}), {})
        self.assertEqual(extract({}), {})

    def test_22_extract_never_raises_for_expected_runtime_failures(self):
        """22. extract() never raises for expected runtime failures."""
        weird_event = {"kind": "chunk", "text": {"nested": [1, 2, 3]}}
        self.assertEqual(extract(weird_event), {})

        with patch.object(perception, "_dispatch_extract", side_effect=Exception("Catastrophic error")):
            self.assertEqual(extract({"kind": "chunk", "text": "hello"}), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
