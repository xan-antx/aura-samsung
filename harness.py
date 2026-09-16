"""Virtual-clock streaming harness + scorer for the Theme 05 agent.

Discrete-event simulation: no real sleeping, no wall-clock flakiness, identical
trace on every run. Replace MOCK_TOOLS and SCENARIOS with the organisers' kit
when it lands; agent.py does not change.

    python harness.py            # run all scenarios, print traces + score
    python harness.py -q         # score only
"""

import heapq
import json
import sys

from agent import Agent, CALL, CANCEL, CLARIFY, FINAL, SAY

# --- tool manifest & mocks -------------------------------------------------

MANIFEST = {
    "search_flights": {"mutating": False, "latency": 0.9},
    "book_flight":    {"mutating": True,  "latency": 1.1},
    "create_ticket":  {"mutating": True,  "latency": 0.8},
}


def run_tool(tool: str, args: dict, attempt: int, faults: set) -> dict:
    if tool in faults and attempt == 1:
        return {"ok": False, "error": "upstream_timeout"}
    if tool == "search_flights":
        return {"ok": True, "flights": [f"{args['destination'][:3].upper()}-101",
                                        f"{args['destination'][:3].upper()}-204"]}
    if tool == "book_flight":
        return {"ok": True, "booking_ref": f"PNR{abs(hash(json.dumps(args, sort_keys=True))) % 10000:04d}"}
    return {"ok": True}


# --- scenarios -------------------------------------------------------------
# events: (t, {...}).  expect: what a correct agent must do.

SCENARIOS = [
    {
        "name": "mid-utterance destination change",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "I need a flight from Delhi to Mumbai tomorrow"}),
            (0.6, {"kind": "interrupt", "text": "sorry, change that to Goa"}),
            (2.4, {"kind": "chunk", "text": "yes book it for two passengers", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "book_flight"},
                   "cancels": 1,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15", "pax": 2, "commit": True},
                   "no_stale_args": [("search_flights", {"destination": "mumbai"})]},
    },
    {
        "name": "duplicate booking guard",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Pune to Chennai on Friday"}),
            (1.5, {"kind": "chunk", "text": "book it"}),
            (3.5, {"kind": "chunk", "text": "did that work, book it again please", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "book_flight"},
                   "cancels": 0,
                   "max_mutating_per_key": 1,
                   "slots": {"origin": "pune", "destination": "chennai",
                             "date": "2026-09-18", "commit": True}},
    },
    {
        "name": "missing slot -> clarify, never guess",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "book me a flight to Bangalore tomorrow", "final": True}),
        ],
        "expect": {"tools_ok": set(), "cancels": 0, "clarify": True,
                   "slots": {"destination": "bengaluru", "date": "2026-09-15",
                             "commit": True}},
    },
    {
        "name": "read-only retry after injected fault",
        "multimodal": False,
        "faults": {"search_flights"},
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Delhi to Goa tomorrow"}),
            (3.0, {"kind": "chunk", "text": "that's all", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights"}, "cancels": 0,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15"}},
    },
    {
        "name": "multimodal: frame grounding behind an acknowledgment",
        "multimodal": True,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "I want to fly tomorrow"}),
            (0.8, {"kind": "frame", "caption": "boarding pass from Delhi to Pune"}),
            (2.5, {"kind": "chunk", "text": "book it", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "book_flight"}, "cancels": 0,
                   "ack_before_perception": True,
                   "slots": {"origin": "delhi", "destination": "pune",
                             "date": "2026-09-15", "commit": True}},
    },
    {
        # The correction lands AFTER the commit keyword. A per-session chunk
        # counter is satisfied by then; only per-slot dwell + a grace window
        # can hold the booking back until the destination stops moving.
        "name": "correction lands after the commit word",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "I need a flight from Delhi to Mumbai tomorrow"}),
            (1.0, {"kind": "chunk", "text": "yes book it"}),
            (1.2, {"kind": "interrupt", "text": "no wait, make that to Goa, sorry"}),
            (2.5, {"kind": "chunk", "text": "that's all", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "book_flight"},
                   "cancels": 0,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15", "commit": True},
                   "never_called_with": [("book_flight", {"flight_id": "MUM-101"})]},
    },
]

SUBSTANTIVE = {SAY, CALL, CLARIFY, FINAL}


# --- simulator -------------------------------------------------------------

def simulate(scn, ticks=True) -> dict:
    """ticks=False replays the stream with no timer support at all, the way a
    fixed-replay eval kit would - correctness must not depend on ticks."""
    agent = Agent(MANIFEST)
    q, seq, trace = [], 0, []
    attempts, cancelled, mutating_keys = {}, set(), []
    ticks_scheduled = set()

    for t, ev in scn["events"]:
        heapq.heappush(q, (t, seq, ev)); seq += 1

    budget = 200                                        # runaway guard: fail loud, not OOM
    while q:
        budget -= 1
        if budget < 0:
            raise RuntimeError(f"{scn['name']}: event budget exhausted (planner loop?)")
        now, _, ev = heapq.heappop(q)
        if ev["kind"] == "tool_result" and ev["call_id"] in cancelled:
            continue                                        # cancelled: never delivered
        trace.append({"dir": "in", "t": now, **ev})

        for a in agent.handle(ev, now):
            trace.append({"dir": "out", **json.loads(a.as_json())})

            if a.kind == CALL:
                tool, args = a.payload["tool"], a.payload["args"]
                if a.payload["mutating"]:
                    mutating_keys.append(a.payload["idem_key"])
                attempts[tool] = attempts.get(tool, 0) + 1
                lat = MANIFEST[tool]["latency"]
                res = run_tool(tool, args, attempts[tool], scn["faults"])
                # a call may be emitted ahead of its release time (a.t > now);
                # the tool runs when the call is released, not when emitted
                heapq.heappush(q, (a.t + lat, seq, {
                    "kind": "tool_result", "call_id": a.payload["call_id"],
                    "tool": tool, "args": args, "result": res})); seq += 1
            elif a.kind == CANCEL:
                cancelled.add(a.payload["call_id"])

        # the agent deferred a mutating call behind a grace window; deliver a
        # tick when it expires (the real adapter maps this to a timer)
        w = agent.next_wakeup()
        if ticks and w is not None and w not in ticks_scheduled:
            ticks_scheduled.add(w)
            heapq.heappush(q, (w, seq, {"kind": "tick"})); seq += 1

    return {"trace": trace, "snapshot": agent.snapshot(),
            "cancelled": cancelled, "mutating_keys": mutating_keys,
            "blocked_duplicates": agent.blocked_duplicates, "agent": agent}


# --- scorer (published rubric: 40 / 35 / 15 / 10) ---------------------------

def score(scn, run) -> tuple[float, list[str]]:
    exp, trace, notes = scn["expect"], run["trace"], []
    outs = [e for e in trace if e["dir"] == "out"]
    # ticks are internal timers, not user inputs - no reply owed to them
    ins = [e for e in trace if e["dir"] == "in" and e["kind"] not in ("tool_result", "tick")]
    ok_tools = {e["tool"] for e in trace
                if e["dir"] == "in" and e["kind"] == "tool_result"
                and e["result"].get("ok") and e["call_id"] not in run["cancelled"]}

    # -- task completion (40)
    tc = 0.0
    if exp["tools_ok"] <= ok_tools:
        tc += 20
    else:
        notes.append(f"missing successful tools {sorted(exp['tools_ok'] - ok_tools)}")
    if run["snapshot"]["slots"] == exp["slots"]:
        tc += 20
    else:
        notes.append(f"snapshot drift: {run['snapshot']['slots']} != {exp['slots']}")
    if exp.get("clarify") and not any(o["kind"] == CLARIFY for o in outs):
        tc -= 20; notes.append("required clarification not asked")
    if not exp.get("clarify") and not any(o["kind"] == FINAL for o in outs):
        tc -= 10; notes.append("turn ended without a final response")

    # -- interruption recovery (35)
    ir = 0.0
    got = sum(1 for o in outs if o["kind"] == CANCEL)
    ir += 20 if got == exp["cancels"] else 0
    if got != exp["cancels"]:
        notes.append(f"cancels {got}, expected {exp['cancels']}")
    stale = [e for e in trace if e["dir"] == "in" and e["kind"] == "tool_result"
             and e["call_id"] in run["cancelled"]]
    if stale:
        notes.append(f"{len(stale)} stale results applied")
    else:
        ir += 15
    # a mutating CALL issued with stale args is a violation even if later
    # cancelled - the irreversible request already left the agent
    for tool, must_not in exp.get("never_called_with", []):
        bad = [o for o in outs if o["kind"] == CALL and o.get("tool") == tool
               and all(o["args"].get(k) == v for k, v in must_not.items())]
        if bad:
            ir -= 15; notes.append(f"mutating call issued on stale slots: {tool} {must_not}")
    for tool, must_not in exp.get("no_stale_args", []):
        bad = [e for e in trace if e["dir"] == "in" and e["kind"] == "tool_result"
               and e["tool"] == tool and e["call_id"] not in run["cancelled"]
               and all(e["args"].get(k) == v for k, v in must_not.items())]
        if bad:
            ir -= 15; notes.append(f"stale re-run reached {tool} with {must_not}")

    # -- latency / floor management (15)
    lat = 15.0
    for e in ins:
        after = [o for o in outs if o["t"] >= e["t"] and o["kind"] in SUBSTANTIVE]
        if not after:
            lat -= 5; notes.append(f"silent after input at t={e['t']}")
    # a barge-in reply replaces speech in progress; it is not an extra filler
    says = [o["t"] for o in outs if o["kind"] == SAY and not o.get("supersedes_speech")]
    spam = sum(1 for a, b in zip(says, says[1:]) if b - a < 1.0)
    if spam:
        lat -= 3 * spam; notes.append(f"{spam} filler(s) inside 1s of the previous one")
    if exp.get("ack_before_perception"):
        frame_t = next(e["t"] for e in ins if e["kind"] == "frame")
        if not any(o["kind"] == SAY and o["t"] == frame_t for o in outs):
            lat -= 5; notes.append("no acknowledgment before perception work")

    # -- safety & protocol (10)
    sf = 0.0
    keys = run["mutating_keys"]
    sf += 5 if len(keys) == len(set(keys)) else 0
    if len(keys) != len(set(keys)):
        notes.append("duplicate state-changing call issued")
    schema_ok = all(isinstance(o.get("t"), (int, float)) and o.get("kind") for o in outs) and \
                all("call_id" in o for o in outs if o["kind"] in (CALL, CANCEL))
    sf += 5 if schema_ok else 0
    if not schema_ok:
        notes.append("malformed action payload")
    for o in outs:
        if o["kind"] == SAY and any(w in o.get("text", "").lower()
                                    for w in ("booked", "done", "confirmed")):
            sf -= 5; notes.append("false completion claim on the fast path")

    total = max(0.0, min(100.0, tc + ir + lat + sf))
    if scn["multimodal"]:
        total = min(100.0, total * 1.5)     # hidden-set multimodal multiplier
    return total, notes


# --- main ------------------------------------------------------------------

def main(quiet=False):
    totals = []
    for scn in SCENARIOS:
        run = simulate(scn)
        pts, notes = score(scn, run)
        totals.append(pts)
        flag = "MM " if scn["multimodal"] else "   "
        print(f"{flag}{pts:6.1f}  {scn['name']}")
        for n in notes:
            print(f"          ! {n}")
        if not quiet:
            for e in run["trace"]:
                if e["dir"] == "in":
                    d = e.get("text") or e.get("caption") or \
                        (f"{e['tool']} {e['result']}" if e["kind"] == "tool_result" else "")
                    print(f"            {e['t']:5.1f} <- {e['kind']:12} {d}")
                else:
                    d = e.get("text") or f"{e.get('tool', e.get('reason',''))} {e.get('args', e.get('invalidated_by',''))}"
                    print(f"            {e['t']:5.1f} -> {e['kind']:12} {d}")
            print()
    print(f"\n{'mean':>8}: {sum(totals)/len(totals):.1f} / 100   over {len(totals)} scenarios")
    return totals


def _selfcheck():
    """Smallest thing that fails if the coordinator breaks."""
    run = simulate(SCENARIOS[0])
    cancels = [e for e in run["trace"] if e["dir"] == "out" and e["kind"] == CANCEL]
    assert len(cancels) == 1, f"expected 1 cancel, got {len(cancels)}"
    assert cancels[0]["invalidated_by"] == ["destination"], cancels[0]
    # the book_flight call survived the destination change and still ran
    assert any(e["dir"] == "in" and e["kind"] == "tool_result" and e["tool"] == "book_flight"
               and e["call_id"] not in run["cancelled"] for e in run["trace"])
    dup = simulate(SCENARIOS[1])
    assert dup["blocked_duplicates"] >= 1, "duplicate booking was not blocked"
    assert len(dup["mutating_keys"]) == len(set(dup["mutating_keys"]))

    # -- stability engine ---------------------------------------------------

    def _out(run, kind, tool=None):
        return [e for e in run["trace"] if e["dir"] == "out" and e["kind"] == kind
                and (tool is None or e.get("tool") == tool)]

    # correction after the commit word: the grace window must swallow the stale
    # booking entirely - no MUM call, no cancel, only the corrected booking
    late = simulate(SCENARIOS[5])
    books = _out(late, CALL, "book_flight")
    assert books, "corrected booking never fired"
    assert all(b["args"]["flight_id"].startswith("GOA") for b in books), books
    assert not late["cancelled"], "engine should defer the booking, not cancel it"
    assert books[0]["t"] >= 1.2 + Agent.GRACE, f"booking fired inside the grace window: {books[0]['t']}"
    # SlotMeta is per slot: destination carries the revision, origin does not
    meta = late["agent"].meta
    assert meta["destination"].revision_count == 2 and meta["destination"].last_changed_at == 1.2
    assert meta["origin"].revision_count == 1
    # the stale Delhi->Mumbai search result was purged when destination moved
    assert all(f.startswith("GOA") for r in late["agent"].results.values()
               for f in r.get("flights", []))

    # a repair cue alone (no slot change) freezes the mutating call past the cue
    cue = {"name": "cue freeze", "multimodal": False, "faults": set(), "events": [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (2.0, {"kind": "chunk", "text": "book it"}),
        (2.2, {"kind": "interrupt", "text": "wait"}),
        (4.0, {"kind": "chunk", "text": "that's all", "final": True}),
    ]}
    run = simulate(cue)
    books = _out(run, CALL, "book_flight")
    assert len(books) == 1, books
    assert books[0]["t"] >= 2.2 + Agent.GRACE, \
        f"repair cue did not freeze the booking: fired at {books[0]['t']}"

    # past the revision cap: clarify instead of a longer wait; the user's answer
    # de-escalates and the booking then fires on the confirmed value
    esc = {"name": "over-revised", "multimodal": False, "faults": set(), "events": [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (0.5, {"kind": "interrupt", "text": "no to Goa"}),
        (1.0, {"kind": "interrupt", "text": "actually to Pune"}),
        (2.5, {"kind": "chunk", "text": "book it", "final": True}),
        (3.0, {"kind": "chunk", "text": "to Pune", "final": True}),
    ]}
    run = simulate(esc)
    clar = [c for c in _out(run, CLARIFY) if "confirm" in c]
    assert clar and "destination" in clar[0]["confirm"], clar
    books = _out(run, CALL, "book_flight")
    assert len(books) == 1 and books[0]["args"]["flight_id"] == "PUN-101", books
    assert books[0]["t"] >= 3.0, "booking fired before the user confirmed"
    assert any(e["kind"] == FINAL and e.get("booked") for e in run["trace"]
               if e["dir"] == "out")

    # ticks are an optimisation, not a requirement: a replayed event stream
    # with no timer support must still complete every booking at full score
    for scn in SCENARIOS:
        no_tick = simulate(scn, ticks=False)
        pts, why = score(scn, no_tick)
        assert pts == 100.0, f"no-tick replay: {scn['name']} scored {pts}: {why}"

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main(quiet="-q" in sys.argv)
