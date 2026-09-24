"""Interruptible real-time agent core.

Design note: the brain is a PURE SYNCHRONOUS function of (event, now) -> [actions].
No asyncio in here. That makes every interruption path deterministically testable,
and the adapter to the organisers' two-async-queue harness is ~20 lines (see README).

The contribution is the Coordinator: dependency-tracked cancellation.
Every call records WHICH SLOTS it was built from - directly, or transitively
when an argument is derived from another call's result (that call's reads are
inherited). When a slot changes mid-utterance we invalidate exactly the work
whose inputs went stale - cancelling it if running, forgetting it if finished -
and let the rest keep running. Cancel-everything loses task completion;
cancel-nothing loses interruption recovery. This wins both.
"""

import hashlib
import json
from dataclasses import dataclass, field

# Perception is the ML boundary (LLM slot extraction, ASR, VLM). The guarded
# import keeps this file pure-stdlib and self-sufficient: if perception.py is
# missing or broken, the keyword fallback below takes over, and with no API
# key configured perception itself degrades to the same deterministic
# extractor - so the harness stays reproducible either way.
try:
    from perception import extract as _perception_extract
except Exception:
    _perception_extract = None

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
    reads: frozenset      # slots this call's arguments were derived from,
                          # including slots inherited through result-derived args
    mutating: bool
    idem_key: str | None
    issued_at: float
    result: dict | None = None    # attached on completion; the call then lives
                                  # on in Agent.done as a dependency-graph node
    superseded: bool = False      # completed mutation whose reads later went
                                  # stale: can't be undone by forgetting, so it
                                  # is disclosed in the final response instead


def _idem(tool: str, args: dict) -> str:
    """Stable key over (tool, args). Two plans that mean the same booking collide."""
    blob = tool + "|" + json.dumps(args, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- slot extraction ------------------------------------------------------

CITIES = {"delhi", "mumbai", "bengaluru", "bangalore", "chennai", "goa", "pune"}
_ALIAS = {"bangalore": "bengaluru"}

# Keyword extractor: the last-resort fallback when perception.py cannot even
# be imported. Deterministic and zero-dependency, mirrored by perception's own
# internal fallback, so all three paths honour the same contract.
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


def _sense(ev: dict) -> dict:
    """The extract(event) -> dict boundary: perception when importable, the
    local keyword matcher when not. Same contract both ways - a partial slot
    dict, {} on any failure, never raises, synchronous."""
    if _perception_extract is not None:
        return _perception_extract(ev)
    return _extract(ev.get("text") or ev.get("caption") or "")


# Deterministic yes/no detection for answers to the agent's own questions -
# coordinator-side, like repair cues, so it works with or without an LLM.
# The vocabulary is deliberately shaped around what the agent asks: the
# confirmation ends "..., correct?", so the literal echo must be understood.
_AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay",
           "correct", "right", "absolutely", "perfect"}
_AFFIRM_PHRASES = (("go", "ahead"), ("do", "it"), ("that's", "right"),
                   ("sounds", "good"), ("please", "do"))
_NEG = {"no", "nope", "nah", "wrong"}
_NEG_PHRASES = (("never", "mind"), ("not", "now"), ("not", "quite"))


def _clean_words(text: str) -> list[str]:
    return [w.strip(".,!?-'") for w in text.lower().replace("’", "'").split()]


def _phrase_in(words: list[str], phrases) -> bool:
    # consecutive cleaned words, never substrings: "cargo ahead" is not
    # "go ahead", and "please don't" is not "please do"
    return any(tuple(words[i:i + len(p)]) == p
               for p in phrases for i in range(len(words) - len(p) + 1))


def _commit_words(text: str) -> bool:
    """The deterministic evidence for a commit - the same rule the keyword
    extractor uses. An LLM may map a bare "yes" to {"commit": true}; without
    an open question or one of these words, that commit is dropped, so a
    stray "yes" can never book anything in either mode."""
    t = text.lower()
    return "book" in t or "confirm" in t


def _negative(text: str) -> bool:
    words = _clean_words(text)
    return bool(words) and (words[0] in _NEG or "don't" in words
                            or _phrase_in(words, _NEG_PHRASES))


def _affirmative(text: str) -> bool:
    if _negative(text):
        return False           # "no, go ahead", "please don't": refusal wins -
                               # a mutation is never executed on a doubtful yes
    words = _clean_words(text)
    return bool(words) and (words[0] in _AFFIRM or _phrase_in(words, _AFFIRM_PHRASES))


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
        self.done: dict[str, Call] = {}         # completed calls, result attached.
                                                # Single source of truth for results,
                                                # their reads, and what's satisfied.
        self.committed: set[str] = set()        # idem keys already executed - a record
                                                # of world changes, never invalidated
        self.retries: dict[str, int] = {}
        self.blocked_duplicates = 0
        self.last_spoke = -99.0
        self.turn_closed = False
        self.commit_at = -1e9                   # last time the user said a commit word
        self.repair_cue_at = -1e9               # last "wait/actually/sorry/no/make that"
        self.wake_at: float | None = None       # deferred mutating call may fire here
        self.deferred: tuple | None = None      # (tool, args, reads, key, gate) held back
        self.pending_q: dict | None = None      # the question on the floor, with its
                                                # exact target: {"kind": "book"|"confirm_slot"
                                                # |"cancel", ...}. One at a time.
        self.declined: set[str] = set()         # question signatures never re-asked
        self.cancel_approved: dict | None = None  # affirmed cancel offer awaiting execution
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
        changed = self._apply(self._gate_commit(_sense(ev), text), now)
        acts = self._invalidate(changed, now)          # cancel stale work FIRST
        acts += self._answer(changed, text, now)       # then resolve the pending question
        acts += self._plan(now)
        if ev.get("final"):
            acts += self._finalise(now)
        return acts

    def _gate_commit(self, extracted: dict, text: str) -> dict:
        """A commit needs evidence: an open question (whose meaning _answer
        owns) or an explicit commit word in the raw text. Otherwise it is an
        extraction artifact of an answer-shaped turn and is dropped. On the
        deterministic path this is a no-op - the keyword extractor only emits
        commit when the words are present."""
        if extracted.get("commit") and self.pending_q is None and not _commit_words(text):
            return {k: v for k, v in extracted.items() if k != "commit"}
        return extracted

    def _answer(self, changed: set, text: str, now: float) -> list[Action]:
        """Resolve the user's turn against the pending question. The decision
        comes from the deterministic yes/no detectors on the RAW TEXT - never
        from what extraction returned, so an LLM that maps "yes" to
        {"commit": true} resolves identically to the keyword path. Runs after
        _apply/_invalidate: a mention of the confirmed slot has already
        resolved the question there, and a real slot change has already
        purged any chain the offer depended on."""
        q = self.pending_q
        if q is None:
            return []                              # a stray "yes" answers nothing
        if _affirmative(text):
            self.pending_q = None
            if q["kind"] == "confirm_slot":
                if self.slots.get(q["slot"]) == q["value"]:
                    self.meta[q["slot"]].revision_count = 1   # confirmed: de-escalate
                    self.commit_at = now           # a fresh go-signal restarts the grace clock
            elif q["kind"] == "book":
                seats = self._latest("check_seat_availability", "seats")
                if seats and seats.result["flight_id"] == q["flight_id"]:
                    # the offered thing IS a commit; the booking still passes
                    # the normal commit gate and grace window in _plan
                    self._apply({"commit": True}, now)
                # else: the offer's chain was invalidated just above - the
                # now-stale offer is dropped and the planner starts over
            elif q["kind"] == "cancel":
                self.commit_at = now
                self.cancel_approved = {"booking_ref": q["booking_ref"]}
            return []
        if _negative(text):
            self.pending_q = None
            self.declined.add(
                f"book:{q['flight_id']}" if q["kind"] == "book"
                else f"cancel:{q['booking_ref']}" if q["kind"] == "cancel"
                else f"confirm:{q['slot']}:{q['value']}")
            if q["kind"] == "confirm_slot":
                # "no" to "is it pune?" means the value is wrong: ask for it
                return [Action(CLARIFY, now, {
                    "text": f"Which {q['slot']} should I use, then?",
                    "missing": [q["slot"]], "state": self.snapshot()})]
            return [self._say("Okay, I'll leave it.", now)]
        if changed - {"commit"}:
            # Supersession needs a REAL slot change (intent counts only when
            # it actually changed; `changed` never holds unchanged keys).
            # Extracting {"commit": true} from an answer is not new content.
            self.pending_q = None
        return []                                  # otherwise the question stays open

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
        changed = self._apply(self._gate_commit(_sense(ev), ev.get("caption", "")), now)
        if changed - {"commit"} and self.pending_q:
            self.pending_q = None                  # new perceived content supersedes it
        acts += self._invalidate(changed, now)
        acts += self._plan(now)
        self.pending_perception -= 1
        return acts

    def _on_result(self, ev, now) -> list[Action]:
        call = self.inflight.pop(ev["call_id"], None)
        if call is None:
            return []                                   # result for a cancelled call: drop
        call.result = ev["result"]
        self.done[call.call_id] = call                  # results go stale like calls do
        if ev["result"].get("ok"):
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
            q = self.pending_q
            if q and q.get("kind") == "confirm_slot" and k == q["slot"]:
                # any answer naming the slot resolves the clarify - a repeat
                # confirms, a new value IS the answer. Either way, de-escalate.
                self.meta[k].revision_count = 1
                self.pending_q = None
        if extracted.get("commit"):
            self.commit_at = now        # every "book it" restarts the grace clock
        return changed

    def _invalidate(self, changed: set, now: float) -> list[Action]:
        """One dependency test for running and finished work alike: anything
        whose reads intersect the change is stale. Running -> cancel. Finished
        read-only -> forget, so no downstream call can build on it. A finished
        MUTATION stays: it changed the world, and the record must survive."""
        if not changed:
            return []
        for cid, call in list(self.done.items()):
            if call.reads & changed:
                if not call.mutating:
                    del self.done[cid]
                elif call.result.get("ok"):
                    # the world already changed; the record stays, but it now
                    # describes something the user no longer asked for. Flag it
                    # so _finalise names it - never silently hold two bookings.
                    call.superseded = True
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

        # Chain link 2: seats for the best flight. flight_id is derived from
        # the search RESULT, so the call inherits the search's reads - a
        # destination change invalidates it even though no slot name appears
        # in its arguments.
        search = self._latest("search_flights", "flights")
        if search:
            acts += self._ensure("check_seat_availability",
                                 {"flight_id": search.result["flights"][0]},
                                 set(search.reads), now)

        # Chain link 3. A state-modifying call only fires once the user has
        # committed, the chain has landed, and the slots have held still.
        # Speculating on a booking is how you double-book. Its reads are the
        # slots in its own args plus everything inherited through the chain.
        seats = self._latest("check_seat_availability", "seats")
        if self.slots.get("commit") and seats:
            reads = {"pax"} | set(seats.reads)
            over = [s for s in sorted(reads)
                    if (m := self.meta.get(s)) and m.revision_count > self.REVISION_CAP]
            if over:
                acts += self._confirm(over[0], now)       # ask; don't wait longer
            else:
                acts += self._ensure("book_flight",
                                     {"flight_id": seats.result["flight_id"],
                                      "seat": seats.result["seats"][0],
                                      "pax": self.slots.get("pax", 1)},
                                     reads, now)

        # An approved cancel offer: fires ONLY on an explicit affirmative to
        # the pending cancel question - never inferred, never automatic. It
        # is mutating, so it rides the same idempotency key and grace gate as
        # every mutation. reads is empty on purpose: it targets a past
        # booking by ref, so no live slot change can invalidate it.
        if self.cancel_approved:
            ref = self.cancel_approved["booking_ref"]
            if any(c.tool == "cancel_booking" and c.result and c.result.get("ok")
                   and c.result.get("cancelled") == ref for c in self.done.values()):
                self.cancel_approved = None            # executed; never twice
            else:
                acts += self._ensure("cancel_booking", {"booking_ref": ref}, set(), now)
        return acts

    def _latest(self, tool: str, field: str) -> Call | None:
        """Most recent surviving completed call of `tool` whose result carries
        `field`. Stale ones were already invalidated, so a survivor is safe to
        derive arguments from - and its reads travel with it."""
        best = None
        for c in self.done.values():
            if c.tool == tool and c.result.get("ok") and c.result.get(field):
                if best is None or c.issued_at > best.issued_at:
                    best = c
        return best

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
        value = self.slots[slot]
        if self.pending_q or f"confirm:{slot}:{value}" in self.declined:
            return []                                    # already asked, or already refused
        self.pending_q = {"kind": "confirm_slot", "slot": slot, "value": value}
        return [Action(CLARIFY, now, {
            "text": f"Just to confirm - {slot} is {value}, correct?",
            "confirm": {slot: value}, "state": self.snapshot()})]

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
        if not mutating and any(c.tool == tool and c.args == args and c.result.get("ok")
                                for c in self.done.values()):
            return []          # already satisfied by a LIVE result; a purged one
                               # no longer satisfies anything, so re-runs are free

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

    # -- turn end ----------------------------------------------------------

    def _finalise(self, now: float) -> list[Action]:
        if self.slots.get("intent") != "flight" and not self.done and not self.inflight:
            # Out of domain: say what we can do and ask, never imply we did
            # something we can't. This is a question, so it goes out as a
            # clarify, not a final.
            return [Action(CLARIFY, now, {
                "text": "I can search for and book flights - that's all I do. "
                        "Where would you like to fly?",
                "missing": ["destination"], "state": self.snapshot()})]

        missing = {"origin", "destination", "date"} - set(self.slots)
        if self.slots.get("intent") == "flight" and missing:
            return [Action(CLARIFY, now, {
                "text": f"Which {sorted(missing)[0]} should I use?",
                "missing": sorted(missing), "state": self.snapshot()})]

        if self.pending_q and self.pending_q["kind"] == "confirm_slot":
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
        bookings = [c for c in self.done.values()
                    if c.mutating and c.result.get("ok") and c.result.get("booking_ref")]
        booked = any(not c.superseded for c in bookings)
        # A final carries its content: name the booking, or name what was
        # found and invite the next step. Completion claims stay tied to a
        # real booking reference (rule 2); the text just stops being empty.
        if booked:
            b = max((c for c in bookings if not c.superseded), key=lambda c: c.issued_at)
            seat = str(b.args.get("seat") or "").split(":")[-1]
            text = (f"You're booked on {b.args['flight_id']}"
                    + (f", seat {seat}" if seat else "")
                    + f" - ref {b.result['booking_ref']}.")
        else:
            search = self._latest("search_flights", "flights")
            seats = self._latest("check_seat_availability", "seats")
            if search:
                flights = search.result["flights"]
                text = f"I found {len(flights)} flights: {', '.join(flights)}."
                if seats:
                    text += f" {len(seats.result['seats'])} seats free on {seats.result['flight_id']}."
                # one question per final, and never re-ask a declined offer
                if f"book:{flights[0]}" not in self.declined:
                    text += f" Want me to book {flights[0]}?"
                    self.pending_q = {"kind": "book", "flight_id": flights[0]}
            elif self.done:
                text = "Here's what I found."
            else:
                text = "I haven't got anything back yet."
        payload = {"text": text, "state": self.snapshot(), "booked": booked}

        cancelled_refs = {c.result.get("cancelled") for c in self.done.values()
                          if c.tool == "cancel_booking" and c.result.get("ok")}
        stale = [c for c in bookings if c.superseded]
        open_stale = [c for c in stale if c.result["booking_ref"] not in cancelled_refs]
        if open_stale:
            # Truthfulness over tidiness: a superseded booking is still real.
            # Name it, and offer the fix only if we can still keep the offer
            # and no other question is already on the floor this turn.
            names = " and ".join(f"{c.args['flight_id']} (ref {c.result['booking_ref']})"
                                 for c in open_stale)
            ref0 = open_stale[0].result["booking_ref"]
            if self.pending_q is None and f"cancel:{ref0}" not in self.declined:
                payload["text"] += f" Your earlier booking {names} is still active - shall I cancel it?"
                self.pending_q = {"kind": "cancel", "booking_ref": ref0,
                                  "flight_id": open_stale[0].args["flight_id"]}
            else:
                payload["text"] += f" Your earlier booking {names} is still active."
            payload["superseded"] = [{"flight_id": c.args["flight_id"],
                                      "booking_ref": c.result["booking_ref"]} for c in open_stale]
        for c in stale:
            if c.result["booking_ref"] in cancelled_refs:
                payload["text"] += (f" Cancelled your earlier booking "
                                    f"{c.args['flight_id']} (ref {c.result['booking_ref']}).")
        return [Action(FINAL, now, payload)]

    def _say(self, text: str, now: float) -> Action:
        self.last_spoke = now
        return Action(SAY, now, {"text": text})
