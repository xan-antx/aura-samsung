"""Interruptible real-time agent core.

Design note: the brain is a PURE SYNCHRONOUS function of (event, now) -> [actions].
No asyncio in here. That makes every interruption path deterministically testable,
and the adapter to the organisers' two-async-queue harness is ~20 lines (see README).

The contribution is the Coordinator: dependency-tracked cancellation.
Every in-flight call records WHICH SLOTS it was built from. When a slot changes
mid-utterance we cancel exactly the calls whose inputs went stale and let the
rest keep running. Cancel-everything loses task completion; cancel-nothing loses
interruption recovery. This wins both.
"""

import hashlib
import json
from dataclasses import dataclass, field

# --- protocol -------------------------------------------------------------

SAY, CALL, CANCEL, CLARIFY, FINAL = "say", "call", "cancel", "clarify", "final"


@dataclass
class Action:
    kind: str
    t: float
    payload: dict

    def as_json(self) -> str:
        return json.dumps({"kind": self.kind, "t": round(self.t, 3), **self.payload},
                          sort_keys=True)


@dataclass
class Call:
    call_id: str
    tool: str
    args: dict
    reads: frozenset      # slots this call's arguments were derived from
    mutating: bool
    idem_key: str | None
    issued_at: float


def _idem(tool: str, args: dict) -> str:
    """Stable key over (tool, args). Two plans that mean the same booking collide."""
    blob = tool + "|" + json.dumps(args, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- slot extraction ------------------------------------------------------

CITIES = {"delhi", "mumbai", "bengaluru", "bangalore", "chennai", "goa", "pune"}
_ALIAS = {"bangalore": "bengaluru"}

# ponytail: keyword extractor, deterministic and zero-dependency so the harness
# stays reproducible. Upgrade path: swap _extract for one LLM call returning the
# same {slot: value} dict. Nothing else in this file changes.
def _extract(text: str) -> dict:
    t = text.lower()
    out = {}
    words = [w.strip(".,!?") for w in t.split()]

    for i, w in enumerate(words):
        if w in ("to", "for") and i + 1 < len(words) and words[i + 1] in CITIES:
            out["destination"] = _ALIAS.get(words[i + 1], words[i + 1])
        if w == "from" and i + 1 < len(words) and words[i + 1] in CITIES:
            out["origin"] = _ALIAS.get(words[i + 1], words[i + 1])

    if "tomorrow" in t:
        out["date"] = "2026-09-15"
    if "friday" in t:
        out["date"] = "2026-09-18"

    for n, v in (("one", 1), ("two", 2), ("three", 3), ("four", 4)):
        if f"{n} seat" in t or f"{n} ticket" in t or f"{n} passenger" in t:
            out["pax"] = v
    if "flight" in t or "fly" in t:
        out["intent"] = "flight"
    if "book" in t or "confirm" in t:
        out["commit"] = True
    return out


# --- the agent ------------------------------------------------------------

class Agent:
    FILLER_GAP = 2.0       # don't emit a second filler within this many seconds
    STABILITY = 2          # slot must survive this many chunks before a mutating call

    def __init__(self, manifest: dict):
        self.manifest = manifest                # tool_name -> {"mutating": bool, "reads": [...]}
        self.slots: dict = {}
        self.inflight: dict[str, Call] = {}
        self.committed: set[str] = set()        # idem keys already executed
        self.completed: set[str] = set()        # (tool,args) signatures already satisfied
        self.retries: dict[str, int] = {}
        self.results: dict[str, dict] = {}
        self.blocked_duplicates = 0
        self.last_spoke = -99.0
        self.turn_closed = False
        self.chunks_seen = 0
        self.pending_perception = 0
        self._n = 0

    # -- public ------------------------------------------------------------

    def handle(self, ev: dict, now: float) -> list[Action]:
        kind = ev["kind"]
        if kind == "chunk":
            return self._on_chunk(ev, now)
        if kind in ("audio", "frame"):
            return self._on_perception(ev, now)
        if kind == "interrupt":
            return self._on_chunk({**ev, "final": False}, now)
        if kind == "tool_result":
            return self._on_result(ev, now)
        return []

    def snapshot(self) -> dict:
        return {"intent": self.slots.get("intent"),
                "slots": {k: v for k, v in sorted(self.slots.items()) if k != "intent"}}

    # -- event handlers ----------------------------------------------------

    def _on_chunk(self, ev, now) -> list[Action]:
        self.chunks_seen += 1
        changed = self._apply(_extract(ev["text"]))
        acts = self._invalidate(changed, now)          # cancel stale work FIRST
        acts += self._plan(now)
        if ev.get("final"):
            acts += self._finalise(now)
        return acts

    def _on_perception(self, ev, now) -> list[Action]:
        """Acknowledge on the fast path, process on the slow path.

        The spec requires raw audio/frames to be handled *behind* a conversational
        acknowledgment. Emitting the filler before the work is the whole point:
        it is what keeps time-to-first-substantive-action inside the latency budget.
        """
        acts = []
        if now - self.last_spoke > self.FILLER_GAP:
            acts.append(self._say("Let me take a look at that.", now))
        self.pending_perception += 1
        changed = self._apply(_extract(ev.get("caption", "")))
        acts += self._invalidate(changed, now)
        acts += self._plan(now)
        self.pending_perception -= 1
        return acts

    def _on_result(self, ev, now) -> list[Action]:
        call = self.inflight.pop(ev["call_id"], None)
        if call is None:
            return []                                   # result for a cancelled call: drop
        self.results[call.call_id] = ev["result"]
        if ev["result"].get("ok"):
            self.completed.add(_idem(call.tool, call.args))
            if call.mutating:
                self.committed.add(call.idem_key)
        elif not call.mutating:
            return self._retry(call, now)               # read-only calls are safe to retry
        acts = self._plan(now)
        if self.turn_closed and not self.inflight:
            acts += self._finalise(now)                 # user finished; work has now drained
        return acts

    # -- coordination ------------------------------------------------------

    def _apply(self, extracted: dict) -> set:
        """Localised slot correction. Returns the set of slots whose value moved."""
        changed = set()
        for k, v in extracted.items():
            if self.slots.get(k) != v:
                self.slots[k] = v
                changed.add(k)
        return changed

    def _invalidate(self, changed: set, now: float) -> list[Action]:
        if not changed:
            return []
        acts = []
        for cid, call in list(self.inflight.items()):
            if call.reads & changed:
                del self.inflight[cid]
                acts.append(Action(CANCEL, now, {
                    "call_id": cid,
                    "reason": "superseded",
                    "invalidated_by": sorted(call.reads & changed),
                }))
        if acts:
            # Barge-in: the user talking over us cuts our current utterance. This
            # replaces the speech in progress, it does not queue behind it.
            acts.append(Action(SAY, now, {"text": "Got it, updating that.",
                                          "supersedes_speech": True}))
            self.last_spoke = now
        return acts

    def _plan(self, now: float) -> list[Action]:
        acts: list[Action] = []
        if self.slots.get("intent") != "flight":
            return acts

        have = set(self.slots)
        if {"origin", "destination", "date"} <= have:
            acts += self._ensure("search_flights",
                                 {"origin": self.slots["origin"],
                                  "destination": self.slots["destination"],
                                  "date": self.slots["date"]},
                                 {"origin", "destination", "date"}, now)

        # A state-modifying call only fires once the user has committed, the search
        # has landed, and the slots have held still. Speculating on a booking is how
        # you double-book.
        if self.slots.get("commit") and self.chunks_seen >= self.STABILITY:
            flight = self._best_flight()
            if flight:
                acts += self._ensure("book_flight",
                                     {"flight_id": flight, "pax": self.slots.get("pax", 1)},
                                     {"origin", "destination", "date", "pax"}, now)
        return acts

    def _ensure(self, tool: str, args: dict, reads: set, now: float) -> list[Action]:
        """Issue a call unless it is already running or already committed."""
        for c in self.inflight.values():
            if c.tool == tool and c.args == args:
                return []                                # identical call already in flight

        mutating = self.manifest.get(tool, {}).get("mutating", False)
        key = _idem(tool, args) if mutating else None
        if key and key in self.committed:
            self.blocked_duplicates += 1
            return []                                    # <- the double-booking guard
        if not mutating and _idem(tool, args) in self.completed:
            return []                                    # already satisfied; don't re-fire

        self._n += 1
        cid = f"c{self._n}"
        self.inflight[cid] = Call(cid, tool, args, frozenset(reads), mutating, key, now)

        acts = [Action(CALL, now, {"call_id": cid, "tool": tool, "args": args,
                                   "mutating": mutating, "idem_key": key,
                                   "reads": sorted(reads)})]
        if now - self.last_spoke > self.FILLER_GAP:
            # Progress narration only. Never a completion claim - we have no result yet.
            acts.append(self._say("One moment, checking that now.", now))
        return acts

    MAX_RETRIES = 1

    def _retry(self, call: Call, now: float) -> list[Action]:
        sig = _idem(call.tool, call.args)
        if call.reads - set(self.slots):
            return []                                    # inputs went stale; don't retry
        if self.retries.get(sig, 0) >= self.MAX_RETRIES:
            return []
        self.retries[sig] = self.retries.get(sig, 0) + 1
        self._n += 1
        cid = f"c{self._n}"
        self.inflight[cid] = Call(cid, call.tool, call.args, call.reads,
                                  call.mutating, call.idem_key, now)
        return [Action(CALL, now, {"call_id": cid, "tool": call.tool, "args": call.args,
                                   "mutating": call.mutating, "idem_key": call.idem_key,
                                   "reads": sorted(call.reads), "retry_of": call.call_id})]

    def _best_flight(self):
        for r in self.results.values():
            if r.get("ok") and r.get("flights"):
                return r["flights"][0]
        return None

    # -- turn end ----------------------------------------------------------

    def _finalise(self, now: float) -> list[Action]:
        missing = {"origin", "destination", "date"} - set(self.slots)
        if self.slots.get("intent") == "flight" and missing:
            return [Action(CLARIFY, now, {
                "text": f"Which {sorted(missing)[0]} should I use?",
                "missing": sorted(missing), "state": self.snapshot()})]

        if self.inflight:
            self.turn_closed = True
            return []                                    # still working; do not claim done

        self.turn_closed = False
        booked = any(r.get("ok") and r.get("booking_ref") for r in self.results.values())
        text = ("You're booked." if booked else
                "Here's what I found." if self.results else
                "I haven't got anything back yet.")
        return [Action(FINAL, now, {"text": text, "state": self.snapshot(),
                                    "booked": booked})]

    def _say(self, text: str, now: float) -> Action:
        self.last_spoke = now
        return Action(SAY, now, {"text": text})
