"""Live chat adapter for Project Aura - the async adapter the README
describes, on a real clock. agent.py is imported, never changed: the brain
stays a pure synchronous function; everything async lives here.

    pip install -r requirements-live.txt
    python live_server.py            # ws://localhost:8765/ws

Per WebSocket session: one Agent, `now` = seconds since session start on the
monotonic loop clock, every handle() call serialised behind one lock. Tool
calls run as asyncio tasks with human-scale latencies (env-configurable) so
a person can interrupt mid-flight; `cancel` actions cancel the task, so a
cancelled call's result is never delivered. `next_wakeup()` maps to a real
timer delivering {"kind": "tick"}, and `t` on call actions is honoured as a
RELEASE time - the grace window is real wall-clock time here.

LLM extraction is on when a provider is configured (AURA_LLM_URL /
OPENAI_API_KEY / GEMINI_API_KEY) - perception reads the env at call time.
Scored harness runs stay hermetic and are untouched by this file.
"""

import asyncio
import datetime
import json
import os

# Live mode resolves relative dates ("tomorrow") against the real calendar;
# scored runs keep the fixed virtual-clock date. Explicit override wins.
os.environ.setdefault("AURA_REF_DATE", datetime.date.today().isoformat())

from aiohttp import web, WSMsgType

import perception
from agent import Agent
from harness import MANIFEST as BASE_MANIFEST, run_tool

# human-scale latencies so interruption is physically possible
LATENCY = {
    "search_flights":          float(os.getenv("LIVE_LAT_SEARCH", "3.5")),
    "check_seat_availability": float(os.getenv("LIVE_LAT_SEATS", "1.5")),
    "book_flight":             float(os.getenv("LIVE_LAT_BOOK", "2.5")),
    "cancel_booking":          float(os.getenv("LIVE_LAT_CANCEL", "1.5")),
}
MANIFEST = {tool: {**spec, "latency": LATENCY.get(tool, spec["latency"])}
            for tool, spec in BASE_MANIFEST.items()}

LLM_ON = any(os.getenv(k) for k in ("AURA_LLM_URL", "OPENAI_API_KEY", "GEMINI_API_KEY"))
PORT = int(os.getenv("LIVE_PORT", "8765"))


class Session:
    """One Agent per WebSocket connection. Session-scoped only."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.out: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.tasks: dict = {}
        self.timers: list = []
        self._fresh()

    def _fresh(self):
        for t in self.tasks.values():
            t.cancel()
        for t in self.timers:
            t.cancel()
        self.tasks, self.timers = {}, []
        self.ticks_scheduled = set()
        self.attempts = {}
        self.agent = Agent(MANIFEST)
        self.t0 = self.loop.time()

    async def reset(self):
        async with self.lock:
            self._fresh()
            while not self.out.empty():          # drop stragglers from the old life
                self.out.get_nowait()
            self.send({"type": "reset", "now": 0.0})

    @property
    def now(self) -> float:
        return self.loop.time() - self.t0

    def send(self, obj):
        self.out.put_nowait(obj)

    async def dispatch(self, ev: dict):
        """Deliver one event to the agent and act on its actions. The lock
        makes every handle() call serial; handle itself runs in a worker
        thread so a slow LLM extraction never stalls timers or the socket."""
        async with self.lock:
            now = self.now
            self.send({"type": "trace", "now": now,
                       "event": {"dir": "in", "t": round(now, 3), **ev}})
            acts = await self.loop.run_in_executor(None, self.agent.handle, ev, now)
            # the status line must say whether LLM calls actually SUCCEED,
            # not merely whether a key is present - push changes as they happen
            state = perception.llm_status()
            if state != getattr(self, "llm_state", None):
                self.llm_state = state
                self.send({"type": "status", "llm": state not in ("off",),
                           "llm_state": state, "now": now, "latencies": LATENCY})
            for a in acts:
                payload = json.loads(a.as_json())
                self.send({"type": "trace", "now": now, "event": {"dir": "out", **payload}})
                if a.kind == "call":
                    self.tasks[payload["call_id"]] = self.loop.create_task(
                        self._run_call(payload))
                elif a.kind == "cancel":
                    task = self.tasks.pop(payload["call_id"], None)
                    if task:
                        task.cancel()            # result never delivered
            w = self.agent.next_wakeup()
            if w is not None and w not in self.ticks_scheduled:
                self.ticks_scheduled.add(w)
                self.timers.append(self.loop.create_task(self._tick_at(w)))

    async def _tick_at(self, w: float):
        await asyncio.sleep(max(0.0, (self.t0 + w) - self.loop.time()))
        await self.dispatch({"kind": "tick"})

    async def _run_call(self, p: dict):
        tool, args, cid = p["tool"], p["args"], p["call_id"]
        # t on a call action is a RELEASE time: hold a deferred mutation until
        # its grace window elapses; a cancel before then retracts it unsent
        await asyncio.sleep(max(0.0, (self.t0 + p["t"]) - self.loop.time()))
        await asyncio.sleep(MANIFEST[tool]["latency"])
        self.attempts[tool] = self.attempts.get(tool, 0) + 1
        result = run_tool(tool, args, self.attempts[tool], set())
        self.tasks.pop(cid, None)
        await self.dispatch({"kind": "tool_result", "call_id": cid,
                             "tool": tool, "args": args, "result": result})


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    sess = Session()
    sess.llm_state = perception.llm_status()
    sess.send({"type": "status", "llm": LLM_ON, "llm_state": sess.llm_state,
               "now": 0.0, "latencies": LATENCY})

    async def drain():
        while True:
            await ws.send_json(await sess.out.get())

    sender = asyncio.create_task(drain())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            if m.get("type") == "chunk" and isinstance(m.get("text"), str):
                await sess.dispatch({"kind": "chunk", "text": m["text"],
                                     "final": bool(m.get("final"))})
            elif m.get("type") == "reset":
                await sess.reset()
    finally:
        sender.cancel()
        sess._fresh()                            # cancel every task and timer
    return ws


def main():
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    print(f"aura live adapter on ws://0.0.0.0:{PORT}/ws  "
          f"(LLM extraction: {'ON' if LLM_ON else 'off - deterministic'})")
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
