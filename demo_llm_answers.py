"""Pending-question answers, verified with an LLM in front of extraction.

Two modes:

  python demo_llm_answers.py          # mock LLM (local, deterministic, no network)
  python demo_llm_answers.py --real   # THE PRE-DEMO GATE against the live model

Mock mode proves the mechanism: a local endpoint that extracts like the
keyword matcher but maps a bare affirmative to {"commit": true} - exactly
what perception's system prompt asks for - must produce action streams
byte-identical to the deterministic path.

--real runs the same flows with whatever provider is configured via
AURA_LLM_URL, OPENAI_API_KEY or GEMINI_API_KEY (refuses to run if none is
set). A real model may legitimately phrase extractions differently, so the
comparison is by OUTCOME, not trace bytes: what got booked, what got
cancelled, which questions were asked, and whether an offer was left open.
Each user turn's real extraction is printed next to the deterministic one.
Exit is non-zero on any outcome mismatch - run this before a live demo.

Deliberately NOT part of `harness.py --selfcheck`, which stays offline and
deterministic. Stdlib only.

Flows:
  1. "yes" to a book offer           -> books exactly the offered flight
  2. "yes" to a slot confirmation    -> unblocks the booking
  3. "yes" to the cancel offer       -> cancels the superseded booking
  4. "book it" DURING a cancel offer -> must NOT cancel anything
  5. "right, change that to Goa"     -> a real slot change supersedes the offer

then every widened detector word ("correct", "right", "yup", "sounds good",
"please do", "absolutely", "perfect", "that's right" / "wrong", "not now",
"not quite") is run as the answer to a live offer.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent import _extract, _clean_words, _AFFIRM
from harness import simulate

LLM_ENV = ("AURA_LLM_URL", "OPENAI_API_KEY", "GEMINI_API_KEY", "AURA_VLM_URL")

FLOWS = [
    ("yes to a book offer", [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Goa tomorrow", "final": True}),
        (2.0, {"kind": "chunk", "text": "yes please", "final": True}),
    ], lambda calls: ("book_flight", "GOA-101") in calls),
    ("yes to a slot confirmation", [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (0.5, {"kind": "interrupt", "text": "no to Goa"}),
        (1.0, {"kind": "interrupt", "text": "actually to Pune"}),
        (2.5, {"kind": "chunk", "text": "book it", "final": True}),
        (3.5, {"kind": "chunk", "text": "yes", "final": True}),
    ], lambda calls: ("book_flight", "PUN-101") in calls),
    ("yes to the cancel offer", [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (1.5, {"kind": "chunk", "text": "book it", "final": True}),
        (5.0, {"kind": "chunk", "text": "wait, to Goa instead", "final": True}),
        (9.0, {"kind": "chunk", "text": "yes", "final": True}),
    ], lambda calls: any(t == "cancel_booking" for t, _ in calls)),
    ("'book it' during a cancel offer must not cancel", [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (1.5, {"kind": "chunk", "text": "book it", "final": True}),
        (5.0, {"kind": "chunk", "text": "wait, to Goa instead", "final": True}),
        (9.0, {"kind": "chunk", "text": "book it", "final": True}),
    ], lambda calls: not any(t == "cancel_booking" for t, _ in calls)),
    ("'right, change that to Goa' supersedes, never confirms", [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow", "final": True}),
        (2.0, {"kind": "chunk", "text": "right, change that to Goa", "final": True}),
    ], lambda calls: not any(t == "book_flight" for t, _ in calls)
                     and ("check_seat_availability", "GOA-101") in calls),
]

BOOK_OFFER = [(0.0, {"kind": "chunk", "text": "flight from Delhi to Goa tomorrow", "final": True})]
AFFIRM_CASES = ["correct", "Correct!", "right", "That's right.", "yup",
                "sounds good", "please do", "absolutely", "Perfect!"]
NEG_CASES = ["wrong", "Not now.", "not quite", "nope"]


class MockLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        slots = _extract(user)                     # extracts like the keyword path...
        words = _clean_words(user)
        if not slots and words and words[0] in _AFFIRM:
            slots["commit"] = True                 # ...but maps a bare affirmative to
                                                   # commit, as perception's prompt
                                                   # instructs. (Whether a real LLM
                                                   # attaches commit to a mixed turn is
                                                   # extraction semantics; this mock
                                                   # isolates answer resolution.)
        resp = json.dumps({"choices": [{"message": {"content": json.dumps(slots)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):
        pass


def env_off():
    return {k: os.environ.pop(k) for k in LLM_ENV if k in os.environ}


def env_restore(saved):
    os.environ.update(saved)


def run(events):
    return simulate({"name": "demo", "multimodal": False, "faults": set(),
                     "events": events})["trace"]


def calls_of(trace):
    return [(e["tool"], e["args"].get("flight_id") or e["args"].get("booking_ref"))
            for e in trace if e.get("dir") == "out" and e.get("kind") == "call"]


def outcome(trace):
    """What actually happened, phrasing-independent: the terms a judge would
    compare - booked what, cancelled what, asked what, ended how."""
    booked, cancelled, ref2flight = [], [], {}
    for e in trace:
        if e.get("dir") == "in" and e.get("kind") == "tool_result" and e["result"].get("ok"):
            if e.get("tool") == "book_flight":
                booked.append(e["args"]["flight_id"])
                ref2flight[e["result"]["booking_ref"]] = e["args"]["flight_id"]
            elif e.get("tool") == "cancel_booking":
                cancelled.append(ref2flight.get(e["result"]["cancelled"], e["result"]["cancelled"]))
    outs = [e for e in trace if e.get("dir") == "out"]
    finals = [o for o in outs if o["kind"] == "final"]
    return {
        "booked": sorted(booked),
        "cancelled": sorted(cancelled),
        "questions_asked": [o["text"] for o in outs if o["kind"] == "clarify"],
        "final_booked": bool(finals and finals[-1].get("booked")),
        "offer_left_open": bool(finals) and finals[-1]["text"].rstrip().endswith("?"),
    }


def main():
    real = "--real" in sys.argv

    if real:
        if not any(os.getenv(k) for k in ("AURA_LLM_URL", "OPENAI_API_KEY", "GEMINI_API_KEY")):
            print("--real needs a configured provider: set AURA_LLM_URL, "
                  "OPENAI_API_KEY or GEMINI_API_KEY. Refusing to run.")
            raise SystemExit(2)
        provider = ("AURA_LLM_URL=" + os.environ["AURA_LLM_URL"] if os.getenv("AURA_LLM_URL")
                    else "OpenAI" if os.getenv("OPENAI_API_KEY") else "Gemini")
        print(f"REAL mode: LLM side uses {provider}; deterministic side runs with "
              "LLM variables stripped. Comparing OUTCOMES, not trace bytes.\n")
        llm_url = None
    else:
        srv = HTTPServer(("127.0.0.1", 0), MockLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        llm_url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
        saved = env_off()
        os.environ["AURA_LLM_URL"] = llm_url
        import perception
        got = perception.extract({"kind": "chunk", "text": "yes"})
        del os.environ["AURA_LLM_URL"]
        env_restore(saved)
        print(f'mock check: "yes" extracts to {got} in LLM mode (vs {{}} deterministically)\n')
        assert got == {"commit": True}, got

    import perception
    ok = True

    def check(name, events, expected, verbose=True):
        nonlocal ok
        saved = env_off()
        det = run(events)                          # deterministic path, always env-free
        if real:
            env_restore(saved)
            llm = run(events)                      # live provider, ambient env
        else:
            os.environ["AURA_LLM_URL"] = llm_url
            llm = run(events)                      # mock endpoint
            del os.environ["AURA_LLM_URL"]
            env_restore(saved)

        behaved = expected(calls_of(llm)) and expected(calls_of(det))
        if real:
            same = outcome(det) == outcome(llm)
        else:
            same = det == llm
        ok &= same and behaved
        mark = "PASS" if same and behaved else "FAIL"

        if real:
            print(f"{mark}  {name}")
            for _, ev in events:                   # what each side extracted, per turn
                text = ev.get("text", "")
                d = _extract(text)
                r = perception.extract({"kind": "chunk", "text": text})
                flag = "" if d == r else "   <- differs"
                print(f'      "{text}"\n            det : {d}\n            real: {r}{flag}')
            if not same:
                print(f"      OUTCOME MISMATCH:\n            det : {outcome(det)}"
                      f"\n            real: {outcome(llm)}")
            elif not behaved:
                print(f"      unexpected calls: {calls_of(llm)}")
            else:
                print(f"      outcome: {outcome(llm)}")
        elif verbose:
            print(f"{mark}  {name}")
            print(f"      traces identical across modes: {det == llm}  ({len(det)} events)")
            print(f"      calls: {calls_of(llm)}")
        else:
            print(f"{mark}  {name}  (identical={det == llm})")

    for name, events, expected in FLOWS:
        check(name, events, expected)

    print("\nwidened detector words against a live book offer:")
    for word in AFFIRM_CASES:
        check(f'affirmative "{word}" books the offer',
              BOOK_OFFER + [(2.0, {"kind": "chunk", "text": word, "final": True})],
              lambda calls: ("book_flight", "GOA-101") in calls, verbose=False)
    for word in NEG_CASES:
        check(f'negative "{word}" declines; stray yes books nothing',
              BOOK_OFFER + [(2.0, {"kind": "chunk", "text": word, "final": True}),
                            (3.5, {"kind": "chunk", "text": "yes", "final": True})],
              lambda calls: not any(t == "book_flight" for t, _ in calls), verbose=False)

    if ok:
        print("\nall flows resolve identically" + (" (by outcome) with the live model"
                                                   if real else " with and without the LLM"))
    else:
        print("\nMISMATCH - do not demo until the above is understood")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
