"""Illustrative demo: pending-question answers resolve identically whether
extraction is the deterministic keyword matcher or an LLM that maps "yes"
to {"commit": true} - the exact behaviour perception's system prompt asks
its model for.

Deliberately NOT part of `harness.py --selfcheck`, which stays offline and
deterministic. Runnable standalone:

    python demo_llm_answers.py

Stdlib only. The mock LLM (OpenAI chat-completions shape, local) extracts
like the keyword matcher, PLUS returns {"commit": true} for affirmatives -
so "yes" comes back as content, not {}. Four flows run under both modes and
their action streams are compared event for event:

  1. "yes" to a book offer         -> books exactly the offered flight
  2. "yes" to a slot confirmation  -> unblocks the booking
  3. "yes" to the cancel offer     -> cancels the superseded booking
  4. "book it" DURING a cancel offer -> must NOT cancel anything
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent import _extract, _clean_words, _AFFIRM
from harness import simulate

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
]


class MockLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        slots = _extract(user)                     # extracts like the keyword path...
        words = _clean_words(user)
        if words and words[0] in _AFFIRM:          # ...but maps affirmatives to commit,
            slots["commit"] = True                 # as perception's prompt instructs
        resp = json.dumps({"choices": [{"message": {"content": json.dumps(slots)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):
        pass


def run(scn_events):
    return simulate({"name": "demo", "multimodal": False, "faults": set(),
                     "events": scn_events})["trace"]


def calls_of(trace):
    return [(e["tool"], e["args"].get("flight_id") or e["args"].get("booking_ref"))
            for e in trace if e.get("dir") == "out" and e.get("kind") == "call"]


def main():
    srv = HTTPServer(("127.0.0.1", 0), MockLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    mock_url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"

    # sanity: with the mock on, "yes" really does extract to {"commit": true}
    os.environ["AURA_LLM_URL"] = mock_url
    import perception
    got = perception.extract({"kind": "chunk", "text": "yes"})
    del os.environ["AURA_LLM_URL"]
    print(f'mock check: "yes" extracts to {got} in LLM mode (vs {{}} deterministically)\n')
    assert got == {"commit": True}, got

    ok = True
    for name, events, expected in FLOWS:
        det = run(events)                          # deterministic path
        os.environ["AURA_LLM_URL"] = mock_url
        llm = run(events)                          # LLM path, same events
        del os.environ["AURA_LLM_URL"]

        identical = det == llm
        behaved = expected(calls_of(llm)) and expected(calls_of(det))
        ok &= identical and behaved
        print(f"{'PASS' if identical and behaved else 'FAIL'}  {name}")
        print(f"      traces identical across modes: {identical}"
              f"  ({len(det)} events)")
        print(f"      calls: {calls_of(llm)}")
    print("\nall flows resolve identically with and without the LLM" if ok
          else "\nMISMATCH - see above")
    srv.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
