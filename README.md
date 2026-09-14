# Theme 05 — Interruptible Real-Time Agent

Dependency-tracked cancellation for full-duplex agents. Pure stdlib, Python 3.10–3.12.

```bash
python harness.py              # all scenarios, full traces, rubric score
python harness.py -q           # scores only
python harness.py --selfcheck  # asserts the coordinator invariants
```

No install step, no dependencies, no network. That satisfies the reproducibility gate on its own.

## The idea

When a user changes their mind mid-utterance, most agents do one of two things:
cancel every in-flight call (safe, but throws away good work and tanks task
completion) or cancel nothing (fast, but acts on stale inputs).

Every call here records **which slots its arguments were derived from**:

```python
Call(tool="search_flights", reads=frozenset({"origin", "destination", "date"}))
```

When a slot moves, we cancel exactly the calls whose `reads` intersect the change
and let the rest run. That's the whole trick, and it's why the same design wins
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

- Chained tool calls where call B's arguments depend on call A's result. The
  `reads` mechanism extends to it (add result-derived slots to `reads`) but it
  isn't wired.
- Unseen tools from a runtime manifest — `MANIFEST` is currently a module constant.
- The quality multiplier (0.80×–1.20×) is not modelled.
