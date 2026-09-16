# Project Aura — Interruptible Real-Time Agent

Samsung PRISM GenAI Hackathon 2026, Theme 05. Team: Thapar_Aura.
Submission 25 Sep 2026. Release tag on final commit: `PRISM_GENAI_HACKATHON_Y2026`.

## What this is

A full-duplex voice agent that survives mid-sentence corrections.

Core idea: every tool call records `reads` — the set of slots its arguments were
derived from, directly or inherited through another call's result. When a slot
changes, invalidate exactly the work whose `reads` intersect the change —
cancel it if running, forget it if finished. Everything else keeps running.
No rollback, so unrelated slots are never lost.

Read `README.md` before changing anything.

## Rules that are not negotiable

1. **Never mutate on a guess.** Read-only calls speculate freely. Mutating calls
   require user commit + slot stability + a grace window.
2. **Never claim completion early.** The fast path narrates progress only.
   "You're booked" requires an actual booking reference to exist.
3. **Never charge twice.** Mutating calls are keyed `sha256(tool, canonical_args)`.
   A committed key cannot be re-issued.
4. **Never go silent.** Every user input produces a substantive action.

If a change would break one of these, stop and say so instead of doing it.

## File ownership — do not edit files you do not own

| File | Owner | Purpose |
|---|---|---|
| `agent.py` | Anant | Coordinator, slots, cancellation, idempotency |
| `perception.py` | ML | ASR, vision, LLM slot extraction |
| `harness.py` | GenAI | Virtual clock, mock tools, scenarios, scorer |
| `ui/` | Full-stack | Live trace visualiser |
| `docker-compose.yml`, `README.md` | Full-stack | Packaging, docs |

If you need a change in someone else's file, say so in the response. Do not
edit across the boundary.

## The two contracts

Everything crosses one of these. Neither signature changes without all four agreeing.

```python
# perception -> agent
extract(event: dict) -> dict          # {"destination": "goa", "pax": 2}
                                      # must return {} on failure, never raise

# agent -> everyone
{"kind": "call" | "cancel" | "say" | "clarify" | "final", "t": float, ...}
```

The UI and the scorer consume the action stream. Neither reads `agent.py`.

## Working style

- Pure stdlib in `agent.py` and `harness.py`. Dependencies go in `perception.py`
  and `ui/` only. A one-command run with no install step is a scored gate.
- `agent.Agent.handle(event, now) -> [Action]` is **pure and synchronous**. No
  asyncio in `agent.py`. This is what makes interruption paths deterministic.
  The async adapter lives in the harness.
- Every new behaviour gets an assertion in `harness.py --selfcheck`. Unasserted
  behaviour is assumed broken.
- Run `python harness.py --selfcheck && python harness.py -q` before every commit.

## Known placeholders

- `_extract` is a keyword matcher. Swap for an LLM call, same return type.
- The scorer implements the published rubric from our reading of the spec. It is
  not Samsung's scorer. Scoring 100 means the agent behaves as designed, nothing more.

(Formerly listed here, now implemented: per-slot stability is
`SlotMeta(last_changed_at, revision_count)` with repair cues, grace-window
deferral, and escalation capped at one step before routing to Clarify — never
scale dwell exponentially, it starves the least certain users. Chained calls
are wired via transitive `reads`: a call built from another call's result
inherits that call's `reads`.)

## Do not

- Add an abstraction with one implementation.
- Rebuild the eval harness once Samsung's kit lands — adapt to theirs.
- Write "guarantees" or "unshakeable" in any doc. The stability engine is an
  estimator with a safety net.
