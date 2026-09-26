import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// Airport vernacular, no dashboard clichés. Every animation corresponds to
// something the agent actually did: a slot flips like a departure board when
// its value really changed, a pulse rides only the dependency edges that
// really carry the change, a grace ring drains only while a mutation is
// really being held. Nothing decorative moves.

const SLOT_KEYS = ["origin", "destination", "date", "pax"];
const SLOT_NAMES = { origin: "Origin", destination: "Destination", date: "Date", pax: "Passengers" };
const STEP_SPEED = 3.5;
const FLAGSHIP = "mid-utterance destination change";
const PARTIAL_PAUSE_MS = 1200;
const GRACE = 0.4;            // agent's hold window, for the draining ring
const RECENT = 0.9;           // how long a change stays "the event on stage"

const RM = (() => {
  try {
    return new URLSearchParams(location.search).has("rm")
      || matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
})();

const PICKER = [
  "mid-utterance destination change",
  "duplicate booking guard",
  "correction arrives after the final marker",
  "rapid double correction escalates to a confirm",
  "multimodal: frame grounding behind an acknowledgment",
];

const AUTHORED_BEATS = [
  { t: 0.3, title: "Searching Delhi → Mumbai",
    caption: "The user is still speaking. We start the search anyway." },
  { t: 0.7, title: "User changes their mind",
    caption: (<>Destination changed. Only calls that read <code>destination</code> are cancelled.</>) },
  { t: 2.0, title: "The rest survives",
    caption: "Origin, date and passenger count were never touched, so no work was wasted." },
  { t: 3.9, title: "Booked on Goa",
    caption: "One booking. One PNR. The Mumbai search never reached a payment." },
];

const cap = (s) => String(s).replace(/\b[a-z]/g, (m) => m.toUpperCase());

function cardTitle(c) {
  const a = c.args || {};
  if (c.tool === "search_flights" && a.destination)
    return `Searching flights ${cap(a.origin ?? "?")} → ${cap(a.destination)}`;
  if (c.tool === "check_seat_availability" && a.flight_id)
    return `Checking seats on ${a.flight_id}`;
  if (c.tool === "book_flight") {
    const seat = (a.seat || "").split(":")[1] || a.seat || "?";
    return `Booking seat ${seat} on ${a.flight_id}` + (a.pax > 1 ? ` for ${a.pax}` : "");
  }
  if (c.tool === "cancel_booking") return `Cancelling booking ${a.booking_ref}`;
  return c.tool;
}

// Mirrors the agent's narration strings exactly, so a progress bubble can be
// linked to its call at render time (same text, same timestamp) - display
// only, nothing is written back anywhere.
function narrOf(c) {
  const a = c.args || {};
  if (c.tool === "search_flights")
    return `Searching flights ${cap(String(a.origin ?? ""))} to ${cap(String(a.destination ?? ""))}...`;
  if (c.tool === "check_seat_availability") return `Checking seats on ${a.flight_id}...`;
  if (c.tool === "book_flight") return `Booking ${a.flight_id}...`;
  if (c.tool === "cancel_booking") return `Cancelling booking ${a.booking_ref}...`;
  return null;
}

function cardResult(r) {
  if (!r) return "";
  if (r.booking_ref) return `Booked, ref ${r.booking_ref}`;
  if (r.cancelled) return `Cancelled ${r.cancelled}`;
  if (r.flights) return `Found ${r.flights.length} flights`;
  if (r.seats) return `${r.seats.length} seats free`;
  return r.ok ? "Done" : r.error || "Failed";
}

function deriveBeats(trace, tEnd) {
  const first = (pred) => trace.find(pred);
  const call = first((e) => e.kind === "call");
  const cancel = first((e) => e.kind === "cancel");
  const clarify = first((e) => e.kind === "clarify");
  const result = first((e) => e.kind === "tool_result");
  const finals = trace.filter((e) => e.kind === "final");
  const closing =
    finals[finals.length - 1] ||
    [...trace].reverse().find((e) => e.kind === "clarify") ||
    trace[trace.length - 1];
  const cand = [];
  if (call) cand.push({ t: call.t, caption: "The first tool call goes out." });
  if (cancel) cand.push({ t: cancel.t, caption: "A call is cancelled — a value it depends on changed." });
  else if (clarify) cand.push({ t: clarify.t, caption: "Instead of guessing, the agent asks." });
  if (result) cand.push({ t: result.t, caption: "The first result comes back and is kept." });
  if (closing)
    cand.push({
      t: closing.t,
      caption:
        closing.kind === "final" ? "The agent gives its final answer."
        : closing.kind === "clarify" ? "Instead of guessing, the agent asks."
        : "The turn ends.",
    });
  const beats = [];
  for (const b of cand.sort((a, c) => a.t - c.t)) {
    const t = Math.min(b.t + 0.1, Math.max(tEnd, 0));
    if (!beats.some((x) => Math.abs(x.t - t) < 0.05)) beats.push({ ...b, t });
  }
  return beats;
}

/* ---- split-flap: one cell per letter, flips only when its letter changes */

function FlapCell({ ch, delay }) {
  const [shown, setShown] = useState(ch);
  const [phase, setPhase] = useState("idle");
  useEffect(() => {
    if (ch === shown) return;
    if (RM) {                          // reduced motion: instant swap
      setShown(ch);
      return;
    }
    const t1 = setTimeout(() => setPhase("out"), delay);
    const t2 = setTimeout(() => { setShown(ch); setPhase("in"); }, delay + 140);
    const t3 = setTimeout(() => setPhase("idle"), delay + 320);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [ch]);                            // eslint-disable-line
  const glyph = shown === " " ? " " : shown;
  // two half-height leaves, each rendering the SAME full glyph at the same
  // size and position - the character reads as one continuous shape with a
  // hinge line drawn over it, never a gap
  return (
    <span className={"flap " + phase}>
      <span className="leaf leaf-t"><i>{glyph}</i></span>
      <span className="leaf leaf-b"><i>{glyph}</i></span>
    </span>
  );
}

function SplitFlap({ value, small }) {
  const v = String(value ?? "—").toUpperCase();
  const [width, setWidth] = useState(v.length);
  useEffect(() => setWidth((w) => Math.max(w, v.length)), [v]);
  // an empty slot is muted text, not a dark tile
  if (v === "—")
    return <span className={"flapboard" + (small ? " small" : "")}>
      <span className="flapempty">—</span>
    </span>;
  const padded = v.padEnd(width, " ");
  // only value.length tiles are visible; trailing pads hold the width but
  // render invisible. Date separators are plain glyphs between tile groups.
  return (
    <span className={"flapboard" + (small ? " small" : "")}>
      {[...padded].map((c, i) =>
        i >= v.length
          ? <span key={i} className="flap pad" aria-hidden="true" />
          : c === "-"
            ? <span key={i} className="flapsep">-</span>
            : <FlapCell key={i} ch={c} delay={i * 55} />)}
    </span>
  );
}

/* ---- grace ring: drains while a mutation is held, never decoration ---- */

function GraceRing({ remain }) {
  const C = 2 * Math.PI * 7;
  return (
    <svg className="ring" viewBox="0 0 18 18" aria-label="grace window">
      <circle cx="9" cy="9" r="7" className="ring-bg" />
      <circle cx="9" cy="9" r="7" className="ring-fg" strokeDasharray={C}
              strokeDashoffset={C * (1 - Math.max(0, Math.min(remain, 1)))}
              transform="rotate(-90 9 9)" />
    </svg>
  );
}

/* ---- boarding pass: a completed booking is a physical object ---- */

function Pass({ c, origin, stamp }) {
  const a = c.args || {};
  const dest = (a.flight_id || "???").split("-")[0];
  const org = (origin || "???").slice(0, 3).toUpperCase();
  const seat = (a.seat || "").split(":")[1] || a.seat || "—";
  return (
    <div className="pass">
      <div className="pass-main">
        <div className="pass-route">
          <b>{org}</b>
          <svg viewBox="0 0 60 16" className="pass-arc">
            <path d="M4 13 Q30 -2 56 13" />
          </svg>
          <b>{dest}</b>
        </div>
        <div className="pass-fields">
          <span>Flight <i>{a.flight_id}</i></span>
          <span>Seat <i>{seat}</i></span>
          <span>Passengers <i>{a.pax ?? 1}</i></span>
        </div>
      </div>
      <div className="pass-stub">
        <span className="pass-ref">{c.result?.booking_ref}</span>
        <span className="pass-bars" aria-hidden="true" />
      </div>
      {stamp && <span className={"stamp " + stamp.kind}>{stamp.text}</span>}
    </div>
  );
}

/* ---- brand mark: a route breaks mid-flight and reroutes; plays once ---- */

function BrandMark() {
  return (
    <span className="brandwrap">
      <svg className="mark" viewBox="0 0 96 30" aria-hidden="true">
        <path className="m-arc1" pathLength="100" d="M5 25 Q30 3 60 17" />
        <path className="m-arc2" pathLength="100" d="M36 12 Q62 1 90 22" />
        <circle className="m-dot" cx="90" cy="22" r="2.4" />
      </svg>
      <span className="brand">Aura</span>
    </span>
  );
}

function App() {
  const [mode, setMode] = useState("replay");
  const [scenarios, setScenarios] = useState([]);
  const [sel, setSel] = useState(0);
  const [view, setView] = useState("timeline");
  const [playhead, setPlayhead] = useState(0);
  const [target, setTarget] = useState(null);
  const rawRef = useRef("");
  const threadRef = useRef(null);

  const [hover, setHover] = useState(null);   // local hover only: {kind, id}
  const [liveTrace, setLiveTrace] = useState([]);
  const [liveStatus, setLiveStatus] = useState({ llm: null, conn: "off", latencies: {} });
  const [livePh, setLivePh] = useState(0);
  const [draft, setDraft] = useState("");
  const wsRef = useRef(null);
  const anchorRef = useRef({ sn: 0, pf: 0 });
  const lastSentRef = useRef("");
  const draftTimerRef = useRef(null);

  useEffect(() => {
    const loadTrace = () => {
      fetch("/trace.json?t=" + Date.now())
        .then((res) => res.json())
        .then((data) => {
          const s = JSON.stringify(data);
          if (s !== rawRef.current) {
            rawRef.current = s;
            const all = data.scenarios || [
              { name: data.scenario || "trace", blurb: "", multimodal: false, trace: data.trace || [] },
            ];
            const offered = PICKER.map((n) => all.find((x) => x.name === n)).filter(Boolean);
            setScenarios(offered.length ? offered : all);
            setSel(0);
            setPlayhead(0);
            setTarget(null);
          }
        })
        .catch((err) => console.error(err));
    };
    loadTrace();
    const interval = setInterval(loadTrace, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (mode !== "live") return;
    const port = new URLSearchParams(location.search).get("live") || 8765;
    const ws = new WebSocket(`ws://${location.hostname}:${port}/ws`);
    wsRef.current = ws;
    setLiveStatus((s) => ({ ...s, conn: "connecting" }));
    ws.onopen = () => setLiveStatus((s) => ({ ...s, conn: "live" }));
    ws.onclose = () => setLiveStatus((s) => ({ ...s, conn: "closed" }));
    ws.onerror = () => setLiveStatus((s) => ({ ...s, conn: "closed" }));
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (typeof m.now === "number")
        anchorRef.current = { sn: m.now, pf: performance.now() };
      if (m.type === "status")
        setLiveStatus((s) => ({
          ...s, llm: m.llm,
          state: m.llm_state || (m.llm ? "untried" : "off"),
          latencies: m.latencies || s.latencies || {},
        }));
      else if (m.type === "reset") {
        setLiveTrace([]);
        lastSentRef.current = "";
      } else if (m.type === "trace") setLiveTrace((tr) => [...tr, m.event]);
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [mode]);

  useEffect(() => {
    if (mode !== "live") return;
    const tick = () => {
      const a = anchorRef.current;
      setLivePh(a.pf ? a.sn + (performance.now() - a.pf) / 1000 : 0);
    };
    const iv = setInterval(tick, 100);
    let raf;
    const smooth = () => {
      tick();
      raf = requestAnimationFrame(smooth);
    };
    raf = requestAnimationFrame(smooth);
    return () => {
      clearInterval(iv);
      cancelAnimationFrame(raf);
    };
  }, [mode]);

  const current = scenarios[sel] || { name: "", blurb: "", multimodal: false, trace: [] };
  const trace = mode === "live" ? liveTrace : current.trace;
  const authored = mode === "replay" && current.name === FLAGSHIP;

  const model = useMemo(() => {
    const calls = [];
    const byId = {};
    const thread = [];
    const slotChanges = [];
    const lastSeen = {};
    let tEnd = 0;
    for (const e of trace) {
      tEnd = Math.max(tEnd, e.t || 0);
      if (e.kind === "call" && e.dir === "out") {
        const c = {
          id: e.call_id, tool: e.tool, args: e.args, mutating: e.mutating,
          reads: e.reads || [],
          start: e.t, end: null, cancelled: false, cancelT: null, by: [], result: null,
          supersededAt: null, cancelledAt: null,
        };
        calls.push(c);
        byId[e.call_id] = c;
        for (const k of SLOT_KEYS)
          if (e.args && e.args[k] !== undefined && lastSeen[k] !== e.args[k]) {
            lastSeen[k] = e.args[k];
            slotChanges.push({ t: e.t, key: k, value: e.args[k] });
          }
      } else if (e.kind === "cancel" && byId[e.call_id]) {
        const c = byId[e.call_id];
        c.cancelled = true;
        c.cancelT = e.t;
        c.end = e.t;
        c.by = e.invalidated_by || [];
      } else if (e.kind === "tool_result" && byId[e.call_id]) {
        byId[e.call_id].end = e.t;
        byId[e.call_id].result = e.result;
      } else if (e.kind === "chunk" || e.kind === "interrupt") {
        thread.push({ t: e.t, side: "user", text: e.text,
                      correction: e.kind === "interrupt",
                      partial: e.kind === "chunk" && !e.final });
      } else if (e.kind === "frame" || e.kind === "audio") {
        thread.push({ t: e.t, side: "user", text: e.caption || "(audio)", frame: true });
      } else if (e.kind === "say" || e.kind === "final" || e.kind === "clarify") {
        thread.push({ t: e.t, side: "agent", text: e.text, kind: e.kind });
      }
    }
    for (const c of calls) if (c.end === null) c.end = tEnd;
    // world truths shared by both views: what got cancelled by the tool, and
    // which completed mutations were later superseded by a slot change
    for (const c of calls) {
      if (c.tool === "cancel_booking" && c.result?.ok) {
        const target = calls.find((b) => b.result?.booking_ref === c.result.cancelled);
        if (target) target.cancelledAt = c.end;
      }
      if (c.mutating && c.result?.ok && !c.cancelled && c.result.booking_ref) {
        const hit = slotChanges.find((s) => s.t > c.end && c.reads.includes(s.key));
        if (hit) c.supersededAt = hit.t;
      }
    }
    return { calls, thread, slotChanges, tEnd };
  }, [trace]);

  const beats = useMemo(
    () => (mode === "live" ? [] : authored ? AUTHORED_BEATS : deriveBeats(trace, model.tEnd)),
    [mode, authored, trace, model.tEnd]
  );

  const tree = useMemo(() => {
    const flat = (v, out = []) => {
      if (v == null) return out;
      if (Array.isArray(v)) v.forEach((x) => flat(x, out));
      else if (typeof v === "object") Object.values(v).forEach((x) => flat(x, out));
      else out.push(String(v));
      return out;
    };
    const calls = model.calls.map((c) => ({ ...c, depth: 1, parent: null }));
    for (const c of calls) {
      const argVals = Object.values(c.args || {}).map(String);
      for (const p of calls) {
        if (p === c || !p.result || p.end > c.start + 1e-9) continue;
        if (p.depth >= c.depth && flat(p.result).some((v) => argVals.includes(v))) {
          c.depth = p.depth + 1;
          c.parent = p.id;
        }
      }
    }
    const maxDepth = Math.max(1, ...calls.map((c) => c.depth));
    const pos = {};
    SLOT_KEYS.forEach((k, i) => {
      pos[k] = { x: ((i + 0.5) / SLOT_KEYS.length) * 100, y: 10 };
    });
    for (let d = 1; d <= maxDepth; d++) {
      const row = calls.filter((c) => c.depth === d).sort((a, b) => a.start - b.start);
      const y = 10 + (d * 80) / Math.max(maxDepth, 2);
      // place each call under the centroid of the slots it reads (fewer
      // crossings), then push overlaps apart preserving order
      const placed = row.map((c) => {
        const xs = c.reads.filter((s) => pos[s]).map((s) => pos[s].x);
        return { c, x: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 50 };
      }).sort((a, b) => a.x - b.x);
      const gap = Math.min(26, 76 / Math.max(placed.length, 1));
      let prev = -100;
      for (const p of placed) {
        p.x = Math.max(p.x, prev + gap);
        prev = p.x;
      }
      const over = placed.length ? placed[placed.length - 1].x - 88 : 0;
      for (const p of placed) {
        if (over > 0) p.x -= over;
        pos[p.c.id] = { x: Math.max(12, Math.min(88, p.x)), y };
      }
    }
    const edges = [];
    for (const c of calls) {
      for (const s of c.reads) if (pos[s]) edges.push({ from: s, to: c.id, slot: s, call: c });
      if (c.parent) edges.push({ from: c.parent, to: c.id, derived: true, call: c });
    }
    return { calls, edges, pos };
  }, [model]);

  const playheadRef = useRef(0);
  playheadRef.current = playhead;
  useEffect(() => {
    if (target === null) return;
    const from = playheadRef.current;
    const dur = (Math.abs(target - from) / STEP_SPEED) * 1000;
    if (dur < 16) {
      setPlayhead(target);
      setTarget(null);
      return;
    }
    const t0 = performance.now();
    let raf;
    const step = (now) => {
      const k = Math.min((now - t0) / dur, 1);
      setPlayhead(from + (target - from) * k);
      if (k >= 1) setTarget(null);
      else raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    const snap = setTimeout(() => {
      setPlayhead(target);
      setTarget(null);
    }, dur + 400);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(snap);
    };
  }, [target]);

  const ph = mode === "live" ? livePh : playhead;
  const beat = beats.reduce((n, b) => (ph >= b.t - 1e-6 ? n + 1 : n), 0);
  const next = () => beat < beats.length && setTarget(beats[beat].t);
  const back = () => setTarget(beat > 1 ? beats[beat - 2].t : 0);

  useEffect(() => {
    if (mode !== "replay") return;
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const seen = (t) => ph >= t - 1e-9;

  // per-slot state at the playhead + which slots changed just now (drives the
  // flap, the pulse and nothing else - motion mirrors the mechanism)
  const slotState = useMemo(() => {
    const st = {};
    SLOT_KEYS.forEach((k) => {
      const past = model.slotChanges.filter((s) => s.key === k && seen(s.t));
      st[k] = {
        cur: past.length ? past[past.length - 1] : null,
        prev: past.length > 1 ? past[past.length - 2] : null,
      };
    });
    return st;
  }, [model, Math.round(ph * 20)]);         // eslint-disable-line
  // only REPLACEMENTS count as "the correction on stage": a slot's first
  // value filling in is the board populating, not a change worth a pulse
  const recentSlots = new Set(
    model.slotChanges
      .filter((s) => ph - s.t >= 0 && ph - s.t < RECENT
        && model.slotChanges.some((p) => p.key === s.key && p.t < s.t))
      .map((s) => s.key)
  );

  const sendChunk = (text, final) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1 && text) ws.send(JSON.stringify({ type: "chunk", text, final }));
  };
  const onDraft = (v) => {
    setDraft(v);
    clearTimeout(draftTimerRef.current);
    draftTimerRef.current = setTimeout(() => {
      const upto = /\s$/.test(v) ? v.trim() : v.slice(0, v.lastIndexOf(" ")).trim();
      if (upto && upto !== lastSentRef.current) {
        lastSentRef.current = upto;
        sendChunk(upto, false);
      }
    }, PARTIAL_PAUSE_MS);
  };
  const onEnter = () => {
    clearTimeout(draftTimerRef.current);
    const full = draft.trim();
    if (!full) return;
    sendChunk(full, true);
    lastSentRef.current = "";
    setDraft("");
  };
  const doReset = () => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "reset" }));
    setLiveTrace([]);
    lastSentRef.current = "";
    setDraft("");
  };

  const actsCC = useMemo(
    () => trace.filter((e) => e.dir === "out" && (e.kind === "call" || e.kind === "cancel")),
    [trace]
  );
  const userEvs = useMemo(() => model.thread.filter((m) => m.side === "user"), [model]);
  const displayThread = useMemo(() => {
    if (mode !== "live") return model.thread;
    return model.thread.filter((m, i) => {
      if (m.side !== "user" || !m.partial) return true;
      return !model.thread.slice(i + 1).some((x) => x.side === "user");
    });
  }, [mode, model]);
  const actedEarly = (m) => {
    if (mode !== "live") return false;
    if (m.partial) return actsCC.some((a) => a.t >= m.t - 1e-9);
    const i = userEvs.indexOf(m);
    if (i > 0 && userEvs[i - 1].partial)
      return actsCC.some((a) => a.t >= userEvs[i - 1].t - 1e-9 && a.t < m.t - 1e-9);
    return false;
  };

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [displayThread.filter((m) => seen(m.t)).length, draft]);

  const banner = authored
    ? beat === 0
      ? { n: null, title: cap(current.name), caption: "Step through with Next." }
      : { n: beat, ...AUTHORED_BEATS[beat - 1] }
    : beat === 0
      ? { n: null, title: cap(current.name), caption: "Step through with Next." }
      : { n: beat, title: current.blurb || cap(current.name), caption: beats[beat - 1].caption };

  const callState = (c) => {
    const dead = c.cancelled && seen(c.cancelT);
    const holding = c.mutating && !seen(c.start) && seen(c.start - GRACE);
    const finished = !c.cancelled && seen(c.end) && c.result;
    return { dead, holding, finished, running: seen(c.start) && !dead && !finished };
  };
  const chipText = (s, c) =>
    s.dead ? "Cancelled" : s.holding ? "Holding" :
    s.finished ? (c.result?.booking_ref ? "Booked" : "Done") : "Running";

  return (
    <div className={"app" + (RM ? " rm" : "")}>
      <header className="top">
        <BrandMark />
        <div className="viewtoggle">
          <button className={mode === "replay" ? "on" : ""} onClick={() => setMode("replay")}>Replay</button>
          <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>Live</button>
        </div>
        {mode === "replay" && (
          <div className="replayctl">
            <select className="picker" value={sel}
                    onChange={(e) => { setSel(Number(e.target.value)); setPlayhead(0); setTarget(null); }}>
              {scenarios.map((s, i) => (
                <option key={s.name} value={i}>
                  {cap(s.name)}{s.multimodal ? "  (multimodal)" : ""}
                </option>
              ))}
            </select>
            <input type="range" min="0" max={model.tEnd || 1} step="0.01" list="beat-ticks"
                   value={Math.min(playhead, model.tEnd || 1)}
                   onChange={(e) => { setTarget(null); setPlayhead(Number(e.target.value)); }} />
            <datalist id="beat-ticks">
              {beats.map((b) => <option key={b.t} value={b.t} />)}
            </datalist>
          </div>
        )}
        <div className="viewtoggle">
          <button className={view === "timeline" ? "on" : ""} onClick={() => setView("timeline")}>Timeline</button>
          <button className={view === "tree" ? "on" : ""} onClick={() => setView("tree")}>Route Map</button>
        </div>
        <div className="controls">
          <span className="clock">t = {ph.toFixed(2)}s</span>
        </div>
      </header>

      {mode === "replay" && current.blurb && <p className="blurb">{current.blurb}</p>}

      {mode === "replay" && (
        <section className="banner">
          <div className="banner-text">
            <h1 className={authored ? "" : "h-blurb"}>
              {banner.n && <span className="beat-n">{banner.n} of {beats.length}</span>}
              {banner.title}
            </h1>
            {banner.caption && <p className="cap">{banner.caption}</p>}
          </div>
          <div className="steps">
            <button className="step back" onClick={back} disabled={beat === 0 && playhead === 0}>← Back</button>
            <button className="step next" onClick={next} disabled={beat === beats.length}>Next →</button>
          </div>
        </section>
      )}

      {mode === "live" && (
        <section className="livebar">
          <span className={"conn " + liveStatus.conn}>
            {liveStatus.conn === "live" ? "Connected" :
             liveStatus.conn === "connecting" ? "Connecting…" : "Disconnected"}
          </span>
          <span className={"llmflag" + (liveStatus.state === "failing" ? " bad" : "")}>
            Extraction: {
              liveStatus.state === "ok" ? "LLM (live)"
              : liveStatus.state === "failing" ? "LLM failing — deterministic fallback"
              : liveStatus.state === "untried" ? "LLM configured, no call yet"
              : liveStatus.state === "off" ? "deterministic (no API key)" : "…"}
          </span>
          <span className="hint">Pause mid-sentence — the agent acts before you press Enter</span>
          <button className="step" onClick={doReset}>Reset</button>
        </section>
      )}

      {/* departure board: the four slots, split-flap. A value flips only
          when the agent's state really changed. */}
      {view === "timeline" && (
        <section className="deck">
          {SLOT_KEYS.map((k) => {
            const { cur } = slotState[k];
            return (
              <div key={k} className={"cell" + (recentSlots.has(k) ? " changed" : "")}>
                <span className="cell-label">{SLOT_NAMES[k]}</span>
                <SplitFlap value={cur === null ? "—" : cur.value} />
              </div>
            );
          })}
        </section>
      )}

      {view === "tree" && (() => {
        // hover focus, derived locally: which edges and nodes belong to the
        // hovered slot or call
        const hlNodes = new Set();
        if (hover?.kind === "slot") {
          hlNodes.add(hover.id);
          tree.calls.forEach((c) => c.reads.includes(hover.id) && hlNodes.add(c.id));
        } else if (hover?.kind === "call") {
          const c = tree.calls.find((x) => x.id === hover.id);
          if (c) {
            hlNodes.add(c.id);
            c.reads.forEach((s) => hlNodes.add(s));
            if (c.parent) hlNodes.add(c.parent);
          }
        }
        const edgeHl = (e) =>
          hover?.kind === "slot" ? (!e.derived && e.slot === hover.id)
          : hover?.kind === "call" ? (e.to === hover.id || (e.derived && e.from === hover.id))
          : false;
        return (
        <main className="treewrap">
          <div className="graph">
            <svg className="wires" viewBox="0 0 100 100" preserveAspectRatio="none">
              {tree.edges.map((e, i) => {
                const a = tree.pos[e.from];
                const b = tree.pos[e.to];
                const c = e.call;
                if (!a || !b || !seen(c.start)) return null;
                const dead = c.cancelled && seen(c.cancelT);
                const HOT = 1.2;
                const hot = dead && ph - c.cancelT < HOT;
                const quiet = dead
                  ? ph - c.cancelT >= HOT
                  : c.result && seen(c.end) && ph - c.end >= HOT;
                const kill = !e.derived && dead && c.by.includes(e.slot);
                // the pulse rides ONLY the changed slot's edges - where it
                // doesn't go is the point
                const pulsing = !e.derived && recentSlots.has(e.slot);
                const cls =
                  "wire" +
                  (e.derived ? " derived" : "") +
                  (kill ? (hot ? " red" : " cooled") : quiet ? " quiet" : "") +
                  (dead && !kill ? " deadedge" : "") +
                  (pulsing ? " pulsing" : "") +
                  (hover ? (edgeHl(e) ? " hl" : " dimmed") : "");
                let y1 = a.y;
                if (!e.derived && slotState[e.slot]?.prev)
                  y1 = a.y + (kill ? -2.4 : 2.4);
                const my = (y1 + b.y) / 2;
                return <path key={i} className={cls} fill="none"
                             d={`M ${a.x} ${y1} C ${a.x} ${my}, ${b.x} ${my}, ${b.x} ${b.y}`} />;
              })}
            </svg>
            {SLOT_KEYS.map((k) => {
              const { cur, prev } = slotState[k];
              const p = tree.pos[k];
              return (
                <div key={k}
                     className={"node slotnode" + (recentSlots.has(k) ? " changed" : "") +
                                (hover ? (hlNodes.has(k) ? " hl" : " dimmed") : "")}
                     style={{ left: `${p.x}%`, top: `${p.y}%` }}
                     onMouseEnter={() => setHover({ kind: "slot", id: k })}
                     onMouseLeave={() => setHover(null)}>
                  <span className="node-key">{SLOT_NAMES[k]}</span>
                  {prev && <span className="node-stale"><s>{String(prev.value)}</s></span>}
                  <SplitFlap small value={cur === null ? "—" : cur.value} />
                </div>
              );
            })}
            {tree.calls.map((c) => {
              const p = tree.pos[c.id];
              const s = callState(c);
              const stale = c.supersededAt !== null && seen(c.supersededAt)
                && !(c.cancelledAt && seen(c.cancelledAt));
              const cls =
                "node callnode" +
                (seen(c.start) || s.holding ? "" : " ghost") +
                (s.dead ? " killed" : stale ? " stale" : s.finished ? " done" : s.running ? " running" : "") +
                (hover ? (hlNodes.has(c.id) ? " hl" : " dimmed") : "");
              const pill = chipText(s, c).toUpperCase();
              return (
                <div key={c.id} className={cls} style={{ left: `${p.x}%`, top: `${p.y}%` }}
                     onMouseEnter={() => setHover({ kind: "call", id: c.id })}
                     onMouseLeave={() => setHover(null)}>
                  <span className="node-title">{cardTitle(c)}</span>
                  <span className="node-status">
                    {s.dead ? (
                      <span className="bad">cancelled — {c.by.map((x) => SLOT_NAMES[x]?.toLowerCase() || x).join(", ")} changed</span>
                    ) : s.holding ? (
                      <>waiting out the grace window <GraceRing remain={(c.start - ph) / GRACE} /></>
                    ) : s.finished ? (
                      <>{cardResult(c.result)}{stale && <b className="still">Still active</b>}</>
                    ) : s.running ? "running…" : "not issued yet"}
                  </span>
                  {(seen(c.start) || s.holding) &&
                    <span className={"pill p-" + pill.toLowerCase()}>{pill}</span>}
                </div>
              );
            })}
            <div className="maplegend">
              <span><svg viewBox="0 0 30 8"><path d="M1 4 H29" className="lg-read" /></svg> Reads this slot</span>
              <span><svg viewBox="0 0 30 8"><path d="M1 4 H29" className="lg-derived" /></svg> Built from that call's result</span>
              <span><svg viewBox="0 0 30 8"><path d="M1 4 H29" className="lg-kill" /></svg> Cancelled by a change</span>
              <span><svg viewBox="0 0 30 8"><path d="M1 4 H29" className="lg-pulse" /></svg> Change in flight</span>
            </div>
          </div>
        </main>
        );
      })()}

      {view === "timeline" && (
        <main className="panels">
          <section className="threadwrap">
            <div className="thread" ref={threadRef}>
              {displayThread.filter((m) => seen(m.t)).map((m, i) => {
                // a progress bubble resolves visually with its call: done
                // becomes a compact check line, cancelled a muted strike
                let progress = null;
                if (m.side === "agent" && m.kind === "say" && /\.\.\.$/.test(m.text || "")) {
                  const c = model.calls.find(
                    (x) => Math.abs(x.start - m.t) < 0.05 && narrOf(x) === m.text);
                  if (c) {
                    if (c.cancelled && seen(c.cancelT)) progress = "dead";
                    else if (seen(c.end) && c.result) progress = "done";
                    else progress = "live";
                  }
                }
                const body =
                  progress === "done" ? "✓ " + m.text.slice(0, -3)
                  : progress === "dead" ? m.text.slice(0, -3)
                  : m.text;
                return (
                <div key={i}
                     className={"msg " + m.side + (m.kind === "say" ? "" : " strong") +
                                (mode === "live" && m.partial ? " partial" : "") +
                                (progress ? " prog-" + progress : "")}>
                  <div className={"bubble" + (m.correction ? " correction" : "") + (m.frame ? " frame" : "")}>
                    {m.correction && <span className="tag">Correction</span>}
                    {m.frame && <span className="tag neutral">Camera</span>}
                    {mode === "live" && m.partial && <span className="tag neutral">Typing…</span>}
                    {actedEarly(m) && <span className="tag early">Acted before Enter</span>}
                    {body}
                  </div>
                  <span className="stamp-t">{m.t.toFixed(1)}s</span>
                </div>
                );
              })}
            </div>
            {mode === "live" && (
              <div className="chatline">
                <input className="chatinput"
                       placeholder={liveStatus.conn === "live" ? "Talk to the agent…" : "Not connected"}
                       value={draft} disabled={liveStatus.conn !== "live"}
                       onChange={(e) => onDraft(e.target.value)}
                       onKeyDown={(e) => e.key === "Enter" && onEnter()} />
                <button className="step next" onClick={onEnter}>Send</button>
              </div>
            )}
          </section>

          <section className="work">
            {model.calls.map((c) => {
              const s = callState(c);
              if (!seen(c.start) && !s.holding) return null;
              const stale = c.supersededAt !== null && seen(c.supersededAt)
                && !(c.cancelledAt && seen(c.cancelledAt));
              const refunded = c.cancelledAt && seen(c.cancelledAt);
              if (s.finished && c.result?.booking_ref) {
                const stamp = refunded ? { kind: "red", text: "Cancelled" }
                  : stale ? { kind: "amber", text: "Still active" } : null;
                return <Pass key={c.id} c={c} origin={slotState.origin.cur?.value} stamp={stamp} />;
              }
              const dur = Math.max(c.end - c.start, 0.02);
              const fill = mode === "live" && s.running && !c.result
                ? Math.min((ph - c.start) / (liveStatus.latencies[c.tool] || 3), 1)
                : Math.max(0, Math.min(ph, c.end) - c.start) / dur;
              // display-only: in Live, a finished call whose slot-named args
              // no longer match the board is visually stale
              const staleLive = mode === "live" && s.finished && !c.cancelled &&
                SLOT_KEYS.some((k) => c.args?.[k] !== undefined
                  && slotState[k].cur && String(slotState[k].cur.value) !== String(c.args[k]));
              const pill = chipText(s, c).toUpperCase();
              return (
                <div key={c.id}
                     className={"strip" + (s.dead ? " killed" : s.finished ? " done" : s.holding ? " holding" : "") +
                                (staleLive ? " stalelive" : "")}>
                  <div className="strip-head">
                    <span className="strip-title">{cardTitle(c)}</span>
                    <span className="pillrow">
                      {staleLive && <span className="pill p-stale">STALE</span>}
                      <span className={"pill p-" + pill.toLowerCase()}>{pill}</span>
                    </span>
                  </div>
                  {s.holding && (
                    <div className="strip-sub">
                      <GraceRing remain={(c.start - ph) / GRACE} />
                      Held for a moment in case you change your mind
                    </div>
                  )}
                  {s.running && !s.holding && (
                    <div className="progress"><i style={{ width: `${Math.min(fill, 1) * 100}%` }} /></div>
                  )}
                  {s.finished && <div className="strip-sub">{cardResult(c.result)}</div>}
                  {s.dead && (
                    <div className="strip-sub bad">
                      Cancelled — {c.by.map((x) => SLOT_NAMES[x]?.toLowerCase() || x).join(", ")} changed
                    </div>
                  )}
                </div>
              );
            })}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
