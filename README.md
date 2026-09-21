# Theme 05 — Interruptible Real-Time Agent

Dependency-tracked cancellation for full-duplex agents. Pure stdlib, Python 3.10–3.12.

```bash
python harness.py              # all scenarios, full traces, rubric score
python harness.py -q           # scores only
python harness.py --selfcheck  # asserts the coordinator invariants
python harness.py --telemetry  # per-scenario latency/cancels/tokens -> telemetry.csv
```

No install step, no dependencies, no network. That satisfies the reproducibility gate on its own.

Or containerised, with the trace visualiser:

```bash
docker compose up --build
```

The `harness` service (stock `python:3.12-slim`, no pip install) runs
`--selfcheck` and then the full scored suite — both stream into the compose
logs — and writes the flagship scenario's trace to `ui/public/trace.json`.
The visualiser starts only after that run completes and serves it at
http://localhost:4173, so the timeline on screen is the run that just
happened on your machine, not a committed snapshot.

## The idea

When a user changes their mind mid-utterance, most agents do one of two things:
cancel every in-flight call (safe, but throws away good work and tanks task
completion) or cancel nothing (fast, but acts on stale inputs).

Every call here records **which slots its arguments were derived from**:

```python
Call(tool="search_flights", reads=frozenset({"origin", "destination", "date"}))
```

An argument derived from another call's *result* inherits that call's `reads`,
so the dependency survives any number of hops — the seat check carries the
search's slots, and the booking carries both:

```python
Call(tool="check_seat_availability",   # args hold no slot at all,
     reads=frozenset({"origin", "destination", "date"}))   # yet it reads these
```

When a slot moves, we invalidate exactly the work whose `reads` intersect the
change — cancelling it if running, forgetting it if finished — and let the rest
run. That's the whole trick, and it's why the same design wins
both Task Completion (40%) and Interruption Recovery (35%) instead of trading one
for the other.

Three supporting guarantees:

- **Idempotency.** State-modifying calls carry `sha256(tool, canonical_args)`. A
  re-plan cannot double-book, because the key is already committed. (Safety, 10%.)
- **No speculative mutation.** Read-only calls fire early and speculatively.
  Mutating calls wait for user commit *and* slot stability. Speculating on a
  booking is how you double-book.
- **No false completion.** The fast path narrates progress only. "You're booked"
  cannot be emitted before a `booking_ref` exists, and the scorer fails the run
  if it is.

## Architecture

The brain is a **pure synchronous function**: `Agent.handle(event, now) -> [Action]`.
No asyncio inside `agent.py`. Every interruption path is deterministically
reproducible, which is why the whole suite runs instantly with no flaky timing.

`harness.py` is a discrete-event simulator over `heapq` — virtual clock, no real
sleeping, identical trace every run.

Completed calls are first-class dependency nodes: a finished call keeps its
`Call` record, result attached, in a single `done` map. That map is the only
source of truth for results, their `reads`, and what counts as already
satisfied. Invalidation is one uniform test — `call.reads & changed` — applied
to in-flight calls (cancel) and completed calls (forget) alike, so the stale
chain downstream of a changed slot dies in a single pass. A completed
**mutation** is exempt: forgetting a booking that already happened can't
un-book it, and would reopen the double-charge path.

## Swapping in the organisers' kit

The eval kit arrives after registration. `agent.py` should not need to change;
write the adapter against the two async queues:

```python
async def adapter(inbox, outbox, manifest):
    agent = Agent(manifest)
    while True:
        ev = await inbox.get()
        for action in agent.handle(ev, ev["t"]):
            await outbox.put(json.loads(action.as_json()))
```

Adapter assumption: `t` on a `call` action is a release time, not an emission
timestamp. If the host executes calls on dequeue, deferred mutating calls must
be held by the adapter until `t` elapses, and a `cancel` arriving before then
retracts them unsent.

`Agent.next_wakeup()` is an optional hint: delivering a `{"kind": "tick"}`
event at that time releases a deferred call promptly. It is purely an
optimisation — any later event flushes the deferral, so a fixed replayed
stream with no timer support is still correct (`--selfcheck` asserts this).

Then replace `MANIFEST`, `run_tool` and `SCENARIOS` in `harness.py` with theirs
and delete the local scorer in favour of the real one.

## Stubbed, deliberately

- `_extract` is a keyword matcher, not an LLM. Deterministic, so the harness
  stays reproducible. Swap it for one LLM call returning the same `{slot: value}`
  dict; nothing else changes.
- Audio and frames arrive pre-captioned. Real ASR/VLM goes behind the same
  acknowledgment-first path that's already wired in `_on_perception`.
- The scorer implements the published rubric (40/35/15/10) from my reading of the
  spec. **It is not the real scorer.** Scoring 100 here means the agent behaves as
  designed, nothing more.

## Not built yet

- Unseen tools from a runtime manifest — `MANIFEST` is currently a module constant.
- The quality multiplier (0.80×–1.20×) is not modelled.
