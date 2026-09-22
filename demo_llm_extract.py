"""Illustrative demo: the LLM extraction path moving a slot the keyword
matcher cannot.

Deliberately NOT part of `harness.py --selfcheck`, which stays offline and
deterministic - this script exists to show the perception boundary working,
not to score anything. Runnable standalone:

    python demo_llm_extract.py

Stdlib only. It spins up its own mock LLM endpoint (OpenAI chat-completions
shape, local, deterministic) and points AURA_LLM_URL at it, then shows the
"hmm, actually let's do Goa instead" case: no "to/for/from <city>" pattern,
so the keyword matcher returns {}, the LLM path returns
{'destination': 'goa'}, and end to end the agent cancels the Mumbai search
and re-issues on Goa.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

TEXT = "hmm, actually let's do Goa instead"   # no 'to/for/from <city>' anywhere


class MockLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"].lower()
        slots = {}
        if "instead" in user and "goa" in user:      # the semantic reading
            slots["destination"] = "goa"
        resp = json.dumps({"choices": [{"message": {"content": json.dumps(slots)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), MockLLM)   # ephemeral port: never collides
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    print(f'utterance          : "{TEXT}"')

    # 1) the keyword matcher alone
    from agent import _extract
    print("keyword matcher    :", _extract(TEXT))

    # 2) perception with the mocked LLM configured
    os.environ["AURA_LLM_URL"] = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    import perception
    print("perception + LLM   :", perception.extract({"kind": "chunk", "text": TEXT}))

    # 3) end to end: the correction cancels the stale search and re-plans
    from agent import Agent
    from harness import MANIFEST
    agent = Agent(MANIFEST)
    acts = agent.handle({"kind": "chunk", "text": "I need a flight from Delhi to Mumbai tomorrow"}, 0.0)
    acts += agent.handle({"kind": "interrupt", "text": TEXT}, 0.6)
    print("\nagent actions:")
    for act in acts:
        print("  ", act.as_json())
    print("\nslots after correction:", agent.snapshot()["slots"])
    srv.shutdown()


if __name__ == "__main__":
    main()
