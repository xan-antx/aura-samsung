"""Virtual-clock streaming harness + scorer for the Theme 05 agent.

Discrete-event simulation: no real sleeping, no wall-clock flakiness, identical
trace on every run. Replace MOCK_TOOLS and SCENARIOS with the organisers' kit
when it lands; agent.py does not change.

    python harness.py            # run all scenarios, print traces + score
    python harness.py -q         # score only
    python harness.py --telemetry   # per-scenario latency/cancels/tokens -> telemetry.csv
    python harness.py --export-trace PATH   # every scenario (name/blurb/score/trace) for
                                            # the visualiser; flagship kept at top level
"""

import hashlib
import heapq
import json
import os
import sys

from agent import Agent, CALL, CANCEL, CLARIFY, FINAL, SAY, _idem

# --- tool manifest & mocks -------------------------------------------------

MANIFEST = {
    "search_flights":          {"mutating": False, "latency": 0.9},
    "check_seat_availability": {"mutating": False, "latency": 0.25},
    "book_flight":             {"mutating": True,  "latency": 1.1},
    "create_ticket":           {"mutating": True,  "latency": 0.8},
}


def run_tool(tool: str, args: dict, attempt: int, faults: set) -> dict:
    # faults entries are either a bare tool name (fails on attempt 1 only,
    # the original semantics every existing scenario relies on) or a
    # (tool, n) pair meaning "fails on every attempt up to and including n" -
    # used to test what happens once the retry budget is actually exhausted.
    for f in faults:
        if isinstance(f, tuple):
            fname, n = f
            if tool == fname and attempt <= n:
                return {"ok": False, "error": "upstream_timeout"}
        elif tool == f and attempt == 1:
            return {"ok": False, "error": "upstream_timeout"}
    if tool == "search_flights":
        return {"ok": True, "flights": [f"{args['destination'][:3].upper()}-101",
                                        f"{args['destination'][:3].upper()}-204"]}
    if tool == "check_seat_availability":
        # seats are flight-scoped: a seat string names the flight it belongs to,
        # so a booking that crosses flights is visible in the trace
        return {"ok": True, "flight_id": args["flight_id"],
                "seats": [f"{args['flight_id']}:12A", f"{args['flight_id']}:14C"]}
    if tool == "book_flight":
        # sha256 over canonical args, like _idem - never built-in hash(), which
        # is seed-randomised per process and would break trace determinism
        return {"ok": True, "booking_ref": f"PNR{int(_idem(tool, args), 16) % 10000:04d}"}
    return {"ok": True}


# --- scenarios -------------------------------------------------------------
# events: (t, {...}).  expect: what a correct agent must do.

SCENARIOS = [
    {
        "name": "mid-utterance destination change",
        "blurb": "The user changes destination mid-sentence; only the stale search is thrown away, everything else keeps running.",
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
        "blurb": "Asked to book the same flight twice, it refuses the second time.",
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
        "blurb": "Told to book with no departure city given, it asks instead of guessing.",
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
        "blurb": "The flight search fails once upstream; it quietly retries and recovers without bothering the user.",
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
        "blurb": "Shown a boarding pass on camera, it says something first, then fills in the trip from the image.",
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
        "blurb": "The correction arrives moments after 'book it' - a short grace window means the wrong flight is never bought.",
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
    {
        # Three-link chain: search -> check_seat_availability(flight_id) ->
        # book_flight(flight_id, seat). The destination changes after link two
        # has COMPLETED. flight_id and seat are derived from results, not from
        # slots, so slot-name reads alone cannot see that the seat belongs to
        # a flight that no longer matches the user's destination.
        "name": "derived args: seat from a superseded flight",
        "blurb": "A seat found on the old flight is never booked: dropping the search also drops everything built on it.",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
            (1.8, {"kind": "chunk", "text": "book it"}),
            (2.0, {"kind": "interrupt", "text": "wait, to Goa instead"}),
            (4.0, {"kind": "chunk", "text": "that's all", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "check_seat_availability", "book_flight"},
                   "cancels": 0,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15", "commit": True},
                   "never_called_with": [("book_flight", {"flight_id": "MUM-101"})],
                   "called_with": [("book_flight", {"flight_id": "GOA-101",
                                                    "seat": "GOA-101:12A"})]},
    },
    {
        # The correction event and the stale call's tool_result share the exact
        # same virtual-clock timestamp (0.9 = search_flights latency). Adversarial
        # timing is explicitly called out in the spec's hidden-set description;
        # this pins the tie-break so it can't regress silently. Scenario events
        # get lower sequence numbers than results generated during the run, so
        # the correction is applied first and the stale result is dropped as
        # cancelled rather than being read at all.
        "name": "interrupt lands at the exact instant a tool returns",
        "blurb": "A correction and a tool result arrive at the very same instant; the correction wins, the stale result is never read.",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
            (0.9, {"kind": "interrupt", "text": "wait, to Goa instead"}),
            (2.5, {"kind": "chunk", "text": "book it", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "book_flight"},
                   "cancels": 1,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15", "commit": True},
                   "never_called_with": [("book_flight", {"flight_id": "MUM-101"})]},
    },
    {
        # The organisers' harness can deliver events after the agent has already
        # emitted a FINAL for a prior turn. A correction here must not roll back
        # a completed booking (it's already real, un-bookable) but it must still
        # be treated as a live instruction, not dropped on the floor.
        "name": "correction arrives after the final marker",
        "blurb": "The user corrects AFTER hearing 'you're booked' - it books the new flight and openly flags the old booking instead of hiding it.",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
            (1.5, {"kind": "chunk", "text": "book it", "final": True}),
            (5.0, {"kind": "chunk", "text": "wait, to Goa instead", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "check_seat_availability", "book_flight"},
                   "cancels": 0,
                   "slots": {"origin": "delhi", "destination": "goa",
                             "date": "2026-09-15", "commit": True},
                   "called_with": [("book_flight", {"flight_id": "GOA-101"})]},
    },
    {
        # Two corrections 20ms apart, no time for anything to resolve between
        # them. Exercises the coordinator's own cancel/replan path independent
        # of the extractor (see the "known gaps" note below the SCENARIOS list) -
        # both hops must cancel cleanly, and hitting REVISION_CAP on the same
        # slot must escalate to a confirm rather than booking on a guess.
        "name": "rapid double correction escalates to a confirm",
        "blurb": "Corrected twice within 20ms, it stops guessing and asks which destination the user actually meant.",
        "multimodal": False,
        "faults": set(),
        "events": [
            (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
            (0.3, {"kind": "interrupt", "text": "no, to Goa"}),
            (0.32, {"kind": "interrupt", "text": "actually, to Pune"}),
            (3.0, {"kind": "chunk", "text": "book it", "final": True}),
        ],
        "expect": {"tools_ok": {"search_flights", "check_seat_availability"},
                   "cancels": 2, "clarify": True,
                   "slots": {"origin": "delhi", "destination": "pune",
                             "date": "2026-09-15", "commit": True}},
    },
]

# --- known gap, not fixed here (out of scope for harness.py) ----------------
# _extract only recognises a destination change when the city is preceded by
# "to"/"for"/"from". A correction phrased WITHOUT that word - "no, Goa",
# "actually Pune", "make that Goa" alone - is silently dropped: no error, no
# clarify, the agent just keeps the old destination and can book the wrong
# city with a clean trace. Confirmed in a scratch run, not encoded as a scored
# scenario here because the fix belongs in perception.py's LLM-based extract
# (the keyword matcher is a documented placeholder, see CLAUDE.md). Flagging
# it so whoever swaps in the real extractor knows to test this phrasing.

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
    for tool, must in exp.get("called_with", []):
        if not any(o["kind"] == CALL and o.get("tool") == tool
                   and all(o["args"].get(k) == v for k, v in must.items()) for o in outs):
            tc -= 10; notes.append(f"required call missing: {tool} {must}")

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
    # A completion claim must not hide an earlier, still-live mutation. From
    # the stream alone, "superseded" means: a successful mutating result that
    # is not the most recent one at FINAL time - the claim is about the latest
    # state, so every older live booking must be named in the spoken text.
    live = [e for e in trace if e["dir"] == "in" and e["kind"] == "tool_result"
            and e["result"].get("ok") and e["result"].get("booking_ref")
            and e["call_id"] not in run["cancelled"]]
    for o in outs:
        if o["kind"] == FINAL and o.get("booked"):
            prior = [e for e in live if e["t"] <= o["t"]]
            for e in prior[:-1]:
                fid = e["args"].get("flight_id", "")
                if fid and fid not in o.get("text", ""):
                    sf -= 5
                    notes.append(f"completion claimed while earlier live booking {fid} is undisclosed")

    total = max(0.0, min(100.0, tc + ir + lat + sf))
    if scn["multimodal"]:
        total = min(100.0, total * 1.5)     # hidden-set multimodal multiplier
    return total, notes


# --- telemetry ---------------------------------------------------------
# Per-scenario metrics for the deck's metrics table: latency, cancel counts,
# and a token-cost figure. This is GenAI's deliverable, feeding Full-stack's
# deck - not agent behaviour, just measurement of it.

def telemetry(scn, run) -> dict:
    trace = run["trace"]
    outs = [e for e in trace if e["dir"] == "out"]
    ins = [e for e in trace if e["dir"] == "in" and e["kind"] not in ("tool_result", "tick")]

    # Same definition of "responded" the scorer uses: time from a real user
    # input to the first substantive (SAY/CALL/CLARIFY/FINAL) output at or
    # after it. This is literally the Response Latency rubric line (15%),
    # just reported per-input instead of collapsed into a single deduction.
    lats = []
    for e in ins:
        after = [o for o in outs if o["t"] >= e["t"] and o["kind"] in SUBSTANTIVE]
        if after:
            lats.append(min(o["t"] for o in after) - e["t"])
    mean_latency = round(sum(lats) / len(lats), 3) if lats else None
    max_latency = round(max(lats), 3) if lats else None

    cancels = sum(1 for o in outs if o["kind"] == CANCEL)

    # Token cost is a PROXY, not a real LLM bill: today's extractor
    # (agent._extract) is a deterministic keyword matcher - zero LLM calls,
    # zero tokens. Once perception.py's LLM-based extract() replaces it, swap
    # this for the actual usage the API reports (prompt + completion tokens
    # per extract() call). Until then, word count of everything the extractor
    # would see stands in for it, so this column - and the deck slot for it -
    # exists now and only the numbers change later, not the column layout.
    words = sum(len((e.get("text") or e.get("caption") or "").split()) for e in ins)

    return {"name": scn["name"], "mean_latency_s": mean_latency,
            "max_latency_s": max_latency, "cancels": cancels,
            "token_cost_proxy": words}


def telemetry_table(write_csv=True) -> list[dict]:
    """Run every scenario, print a metrics table, and (by default) write
    telemetry.csv next to harness.py - Full-stack reads that file for the
    deck's metrics table, no need to touch this file."""
    rows = [telemetry(scn, simulate(scn)) for scn in SCENARIOS]

    print(f"{'scenario':48} {'mean_lat':>9} {'max_lat':>9} {'cancels':>8} {'tokens*':>8}")
    for r in rows:
        ml = f"{r['mean_latency_s']:.3f}" if r["mean_latency_s"] is not None else "-"
        xl = f"{r['max_latency_s']:.3f}" if r["max_latency_s"] is not None else "-"
        print(f"{r['name'][:48]:48} {ml:>9} {xl:>9} {r['cancels']:>8} {r['token_cost_proxy']:>8}")
    print("\n* token_cost_proxy is a word-count stand-in until the LLM extractor lands (see docstring)")

    if write_csv:
        import csv
        with open("telemetry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "mean_latency_s", "max_latency_s",
                                              "cancels", "token_cost_proxy"])
            w.writeheader()
            w.writerows(rows)
        print("\nwrote telemetry.csv")
    return rows


def export_payload() -> dict:
    """Every scenario, in SCENARIOS order, each with its name, blurb,
    multimodal flag, score and full trace. The flagship's name and trace stay
    at the top level under the original "scenario"/"trace" keys - an additive
    superset, so an existing reader of the single-scenario shape (the
    visualiser reads data.trace from a static file and cannot pass a flag)
    keeps working unchanged."""
    entries = []
    for scn in SCENARIOS:
        run = simulate(scn)
        pts, _ = score(scn, run)
        entries.append({"name": scn["name"], "blurb": scn["blurb"],
                        "multimodal": scn["multimodal"], "score": pts,
                        "trace": run["trace"]})
    return {"scenario": entries[0]["name"], "trace": entries[0]["trace"],
            "scenarios": entries}


def export_trace(path):
    """Write the full export where the visualiser serves it. The sim is
    deterministic, so a judge's run produces byte-identical output every time
    (newline="\\n" keeps Windows and container writes identical too)."""
    payload = export_payload()
    with open(path, "w", newline="\n") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path} ({len(payload['scenarios'])} scenarios; "
          f"top-level trace: {payload['scenario']})")


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


# sha256 of the serialized export - see the determinism pin in _selfcheck
EXPORT_DIGEST = "197292b9d504b6f40dd5838ad9af23b221df962010b420fc8d11d8a6ce4c2549"


def _selfcheck():
    """Smallest thing that fails if the coordinator breaks."""
    # determinism is only meaningful offline: no LLM variable may be live here
    assert not any(os.getenv(k) for k in _LLM_ENV), \
        "selfcheck must run hermetic - LLM env vars should have been bypassed"
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
    assert all(f.startswith("GOA") for c in late["agent"].done.values()
               for f in c.result.get("flights", []))

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

    # -- transitive reads through derived arguments ---------------------------

    # a call whose args come from another call's result inherits that call's
    # reads, so one destination change invalidates the whole chain downstream
    chain = simulate(SCENARIOS[6])
    seat_calls = _out(chain, CALL, "check_seat_availability")
    assert seat_calls and all(sc["reads"] == ["date", "destination", "origin"]
                              for sc in seat_calls), \
        "seat check must inherit the search's reads, not its visible slot names"
    books = _out(chain, CALL, "book_flight")
    assert len(books) == 1 and books[0]["args"]["flight_id"] == "GOA-101" \
        and books[0]["args"]["seat"] == "GOA-101:12A", books
    assert books[0]["reads"] == ["date", "destination", "origin", "pax"], \
        "booking must carry both hops of inherited reads plus its own slot"
    assert not chain["cancelled"], "chain died after completion; nothing to cancel"
    # the MUM seat-check result died WITH the destination - two hops from the
    # slot, with no result-specific special case in the coordinator
    assert all(not c.result["flight_id"].startswith("MUM")
               for c in chain["agent"].done.values()
               if c.tool == "check_seat_availability")

    # purge-as-consequence: an invalidated result no longer satisfies anything,
    # so reverting to an earlier value re-runs the search instead of dead-ending
    rev = {"name": "revert", "multimodal": False, "faults": set(), "events": [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Mumbai tomorrow"}),
        (1.5, {"kind": "interrupt", "text": "actually to Goa"}),
        (3.0, {"kind": "interrupt", "text": "sorry, back to Mumbai"}),
        (5.0, {"kind": "chunk", "text": "that's all", "final": True}),
    ]}
    run = simulate(rev)
    mum_searches = [s for s in _out(run, CALL, "search_flights")
                    if s["args"]["destination"] == "mumbai"]
    assert len(mum_searches) == 2, "purged search must be re-runnable after a revert"

    # a superseded completed mutation is disclosed, never forgotten: after the
    # post-final correction, the Goa final names the still-live Mumbai booking
    twice = simulate(SCENARIOS[8])
    finals = _out(twice, FINAL)
    assert len(finals) == 2 and finals[0]["booked"] and finals[1]["booked"]
    assert "superseded" not in finals[0], "nothing to disclose at the first final"
    assert "MUM-101" in finals[1]["text"], finals[1]["text"]
    assert [b["flight_id"] for b in finals[1]["superseded"]] == ["MUM-101"]
    # and the record itself survives in done - the ledger and the disclosure agree
    assert any(c.mutating and c.superseded and c.args["flight_id"] == "MUM-101"
               and c.result["booking_ref"] for c in twice["agent"].done.values())

    # the exported trace is JSON-clean and in the exact shape the visualiser
    # fetches: {"trace": [...]} with every entry serialisable as-is
    payload = json.loads(json.dumps({"trace": simulate(SCENARIOS[0])["trace"]}))
    assert payload["trace"] and all("t" in e and "kind" in e for e in payload["trace"])
    # a literal booking ref only holds if it is derived process-independently
    # (sha256, not seed-randomised hash()) - this fails on the next run if
    # trace determinism regresses
    refs = [e["result"]["booking_ref"] for e in payload["trace"]
            if e.get("kind") == "tool_result" and e["result"].get("booking_ref")]
    assert refs == ["PNR7240"], f"booking_ref not deterministic across processes: {refs}"

    # the full export: ordered like SCENARIOS, blurbed, scored, and the
    # flagship duplicated at the top level for existing readers
    exp = export_payload()
    assert [e["name"] for e in exp["scenarios"]] == [s["name"] for s in SCENARIOS]
    assert exp["scenario"] == SCENARIOS[0]["name"] and exp["trace"] == exp["scenarios"][0]["trace"]
    assert all(e["blurb"] and e["score"] == 100.0 and e["trace"] for e in exp["scenarios"])
    # determinism pin: this literal digest only keeps matching if the export
    # is byte-identical across processes. If you changed a scenario or blurb
    # on purpose, rerun and update the literal; if you changed nothing and
    # this fails, you introduced nondeterminism.
    digest = hashlib.sha256(json.dumps(exp, indent=2).encode()).hexdigest()
    assert digest == EXPORT_DIGEST, f"export digest drifted: {digest}"

    # ticks are an optimisation, not a requirement: a replayed event stream
    # with no timer support must still complete every booking at full score
    for scn in SCENARIOS:
        no_tick = simulate(scn, ticks=False)
        pts, why = score(scn, no_tick)
        assert pts == 100.0, f"no-tick replay: {scn['name']} scored {pts}: {why}"

    # -- retry budget across turns: pinned, not endorsed --------------------
    # MAX_RETRIES=1 caps automatic back-to-back retries fired from _on_result.
    # It does NOT cap attempts overall: any later chunk re-enters _plan, which
    # re-issues the call fresh (nothing in _ensure checks the retries counter,
    # only _retry does). So a tool that fails on every attempt still gets
    # retried again the next time the user says anything at all - here, a
    # third attempt succeeds after two straight failures. This assertion pins
    # today's behaviour so a future change to the policy shows up as a diff
    # here rather than silently, not a claim that 3 attempts is the right cap.
    # Raise with Anant: is this the intended resilience, or should a call that
    # has burned its retry budget stay dead until its own slots change again?
    persistent = {"name": "persistent failure exhausts the retry budget",
                  "multimodal": False, "faults": {("search_flights", 2)}, "events": [
        (0.0, {"kind": "chunk", "text": "flight from Delhi to Goa tomorrow"}),
        (4.0, {"kind": "chunk", "text": "that is all", "final": True}),
    ]}
    run = simulate(persistent)
    attempts = _out(run, CALL, "search_flights")
    assert len(attempts) == 3, \
        f"expected 3 search_flights attempts (1 + 1 retry + 1 re-plan on the next chunk), got {len(attempts)}"
    assert any(e["dir"] == "in" and e["kind"] == "tool_result" and e["tool"] == "search_flights"
               and e["result"].get("ok") for e in run["trace"]), \
        "the third attempt should still succeed once the fault window passes"

    print("selfcheck ok")


# Scored runs are hermetic by contract: an exported API key would route
# extraction through a live LLM and silently break identical-trace replay.
# Scrubbed only when harness.py runs as a script - importing this module
# (the demo, an adapter) leaves the environment alone, and perception.py's
# own behaviour when called directly is unchanged.
_LLM_ENV = ("AURA_LLM_URL", "OPENAI_API_KEY", "GEMINI_API_KEY", "AURA_VLM_URL")


def _force_hermetic():
    bypassed = [k for k in _LLM_ENV if os.environ.pop(k, None) is not None]
    if bypassed:
        print(f"hermetic run: ignoring {', '.join(bypassed)} - "
              "scored runs always use the deterministic extractor")


if __name__ == "__main__":
    _force_hermetic()
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--telemetry" in sys.argv:
        telemetry_table()
    elif "--export-trace" in sys.argv:
        export_trace(sys.argv[sys.argv.index("--export-trace") + 1])
    else:
        main(quiet="-q" in sys.argv)
