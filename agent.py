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
class SlotMeta:
    """Per-slot stability record. Dwell is measured from last_changed_at, not
    from a session-wide chunk count - a slot written 100ms ago is not stable
    no matter how many chunks the session has seen."""
    last_changed_at: float
    revision_count: int = 1


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


REPAIR_CUES = {"actually", "wait", "sorry", "no"}


def _repair_cue(text: str) -> bool:
    """Disfluency on the transcript tail: a correction is probably coming."""
    tail = [w.strip(".,!?-") for w in text.lower().split()][-8:]
    return bool(REPAIR_CUES & set(tail)) or "make that" in " ".join(tail)


# --- the agent ------------------------------------------------------------

class Agent:
    FILLER_GAP = 2.0       # don't emit a second filler within this many seconds
    GRACE = 0.4            # dwell before an irreversible call; only the call waits,
                           # the fast path keeps speaking, so latency-to-speech is 0
    ESCALATED = 0.8        # one escalation step after a correction. Never more:
                           # exponential dwell starves the least certain users
    REVISION_CAP = 2       # revisions beyond this route to Clarify, not longer waits

    def __init__(self, manifest: dict):
        self.manifest = manifest                # tool_name -> {"mutating": bool, "reads": [...]}
        self.slots: dict = {}
        self.meta: dict[str, SlotMeta] = {}     # per-slot stability, updated in _apply
        self.inflight: dict[str, Call] = {}
        self.committed: set[str] = set()        # idem keys already executed
        self.completed: set[str] = set()        # (tool,args) signatures already satisfied
        self.retries: dict[str, int] = {}
        self.results: dict[str, dict] = {}
        self.result_reads: dict[str, frozenset] = {}  # completed results are slot-dependent too
        self.blocked_duplicates = 0
        self.last_spoke = -99.0
        self.turn_closed = False
        self.commit_at = -1e9                   # last time the user said a commit word
        self.repair_cue_at = -1e9               # last "wait/actually/sorry/no/make that"
        self.wake_at: float | None = None       # deferred mutating call may fire here
        self.deferred: tuple | None = None      # (tool, args, reads, key, gate) held back
        self.pending_confirm: str | None = None # slot past the revision cap, awaiting answer
        self.pending_perception = 0
        self._n = 0

    # -- public ------------------------------------------------------------

    def handle(self, ev: dict, now: float) -> list[Action]:
        kind = ev["kind"]
        # Flush first: a deferred call whose grace window has elapsed fires on
        # ANY event. Ticks are an optimisation (they release the call as early
        # as possible); a replayed stream with no timers is still correct.
        acts: list[Action] = []
        if self.wake_at is not None and now >= self.wake_at:
            acts += self._plan(now)
            if self.turn_closed:
                acts += self._finalise(now)
        if kind == "chunk":
            return acts + self._on_chunk(ev, now)
        if kind in ("audio", "frame"):
            return acts + self._on_perception(ev, now)
        if kind == "interrupt":
            return acts + self._on_chunk({**ev, "final": False}, now)
        if kind == "tool_result":
            return acts + self._on_result(ev, now)
        return acts                     # "tick": the flush above was its whole job

    def next_wakeup(self) -> float | None:
        """Optional adapter hint: if set, delivering a {"kind": "tick"} event at
        this time releases a deferred call promptly. Purely an optimisation -
        any later event flushes it, and end-of-turn emits it self-scheduled."""
        return self.wake_at

    def snapshot(self) -> dict:
        return {"intent": self.slots.get("intent"),
                "slots": {k: v for k, v in sorted(self.slots.items()) if k != "intent"}}

    # -- event handlers ----------------------------------------------------

    def _on_chunk(self, ev, now) -> list[Action]:
        text = ev.get("text", "")
        if _repair_cue(text):
            self.repair_cue_at = now               # freeze mutations: correction incoming
        changed = self._apply(_extract(text), now)
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
        changed = self._apply(_extract(ev.get("caption", "")), now)
        acts += self._invalidate(changed, now)
        acts += self._plan(now)
        self.pending_perception -= 1
        return acts

    def _on_result(self, ev, now) -> list[Action]:
        call = self.inflight.pop(ev["call_id"], None)
        if call is None:
            return []                                   # result for a cancelled call: drop
        self.results[call.call_id] = ev["result"]
        self.result_reads[call.call_id] = call.reads    # results go stale like calls do
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

    def _apply(self, extracted: dict, now: float) -> set:
        """Localised slot correction. Returns the set of slots whose value moved."""
        changed = set()
        for k, v in extracted.items():
            if self.slots.get(k) != v:
                self.slots[k] = v
                changed.add(k)
                m = self.meta.get(k)
                if m is None:
                    self.meta[k] = SlotMeta(last_changed_at=now)
                else:
                    m.last_changed_at = now
                    m.revision_count += 1
            if k == self.pending_confirm:
                # any answer naming the slot resolves the clarify - a repeat
                # confirms, a new value IS the answer. Either way, de-escalate.
                self.meta[k].revision_count = 1
                self.pending_confirm = None
        if extracted.get("commit"):
            self.commit_at = now        # every "book it" restarts the grace clock
        return changed

    def _invalidate(self, changed: set, now: float) -> list[Action]:
        if not changed:
            return []
        # completed results derived from a moved slot are stale too - keeping
        # them is how a re-plan books the OLD destination from the old search
        for cid in [c for c, r in self.result_reads.items() if r & changed]:
            del self.result_reads[cid]
            self.results.pop(cid, None)
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
        self.wake_at = self.deferred = None    # recomputed below if still deferring
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
        if self.slots.get("commit"):
            flight = self._best_flight()
            if flight:
                reads = {"origin", "destination", "date", "pax"}
                over = [s for s in sorted(reads)
                        if (m := self.meta.get(s)) and m.revision_count > self.REVISION_CAP]
                if over:
                    acts += self._confirm(over[0], now)   # ask; don't wait longer
                else:
                    acts += self._ensure("book_flight",
                                         {"flight_id": flight, "pax": self.slots.get("pax", 1)},
                                         reads, now)
        return acts

    def _gate(self, reads: set) -> float:
        """Earliest time a mutating call over these slots may fire: a grace
        window after the commit word, the last repair cue, and each read slot's
        last change. One doubled step for a corrected slot, then the cap."""
        gate = max(self.commit_at, self.repair_cue_at) + self.GRACE
        for s in reads:
            m = self.meta.get(s)
            if m is not None:
                dwell = self.ESCALATED if m.revision_count >= 2 else self.GRACE
                gate = max(gate, m.last_changed_at + dwell)
        return gate

    def _confirm(self, slot: str, now: float) -> list[Action]:
        """Past the escalation cap the user is telling us they're not sure.
        A longer dwell would punish exactly them - clarify instead."""
        if self.pending_confirm:
            return []                                    # already on the floor
        self.pending_confirm = slot
        return [Action(CLARIFY, now, {
            "text": f"Just to confirm - {slot} is {self.slots[slot]}, correct?",
            "confirm": {slot: self.slots[slot]}, "state": self.snapshot()})]

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

        if mutating:
            gate = self._gate(reads)
            if now < gate:
                # Grace window. The fast path has already spoken; only the
                # irreversible call waits, so time-to-first-speech is untouched.
                self.wake_at = gate if self.wake_at is None else min(self.wake_at, gate)
                self.deferred = (tool, args, reads, key, gate)
                return []

        acts = self._issue(tool, args, reads, mutating, key, now)
        if now - self.last_spoke > self.FILLER_GAP:
            # Progress narration only. Never a completion claim - we have no result yet.
            acts.append(self._say("One moment, checking that now.", now))
        return acts

    def _issue(self, tool, args, reads, mutating, key, t: float) -> list[Action]:
        """Register the call and emit it stamped at t. t may be in the future:
        the adapter releases it when the clock gets there, and a cancel emitted
        before then retracts it unsent."""
        self._n += 1
        cid = f"c{self._n}"
        self.inflight[cid] = Call(cid, tool, args, frozenset(reads), mutating, key, t)
        return [Action(CALL, t, {"call_id": cid, "tool": tool, "args": args,
                                 "mutating": mutating, "idem_key": key,
                                 "reads": sorted(reads)})]

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

        if self.pending_confirm:
            return []                                    # a clarify is already on the floor

        if self.deferred is not None:
            # End of turn with a call still inside its grace window and no
            # guarantee of further events (or ticks). Emit it now, stamped at
            # its gate time - it fires when the clock elapses, tick or no tick.
            tool, args, reads, key, gate = self.deferred
            self.wake_at = self.deferred = None
            self.turn_closed = True
            return self._issue(tool, args, reads, True, key, gate)

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
