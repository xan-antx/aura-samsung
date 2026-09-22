import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// Two panels, one clock. Left: the conversation as a chat thread - time
// flows downward, so speaking order is unambiguous. Right: one card per
// tool call. The pairing is the point: when the correction message appears
// on the left, the card it invalidates dies on the right at the same
// moment, while unrelated cards keep running.

const SLOT_KEYS = ["origin", "destination", "date", "pax"];
const SLOT_NAMES = { origin: "origin", destination: "destination", date: "date", pax: "passengers" };
const STEP_SPEED = 3.5; // trace-seconds per wall-second while stepping
const FLAGSHIP = "mid-utterance destination change";

// Offered in the picker. The export still carries every scenario; these are
// the five with something to watch.
const PICKER = [
  "mid-utterance destination change",
  "duplicate booking guard",
  "correction arrives after the final marker",
  "rapid double correction escalates to a confirm",
  "multimodal: frame grounding behind an acknowledgment",
];

const AUTHORED_BEATS = [
  {
    t: 0.3,
    title: "Searching Delhi → Mumbai",
    caption: "The user is still speaking. We start the search anyway.",
  },
  {
    t: 0.7,
    title: "User changes their mind",
    caption: (
      <>
        Destination changed. Only calls that read <code>destination</code> are cancelled.
      </>
    ),
  },
  {
    t: 2.0,
    title: "The rest survives",
    caption: "Origin, date and passenger count were never touched, so no work was wasted.",
  },
  {
    t: 3.9,
    title: "Booked on Goa",
    caption: "One booking. One PNR. The Mumbai search never reached a payment.",
  },
];

const cap = (s) => String(s).replace(/\b[a-z]/g, (m) => m.toUpperCase());

// Plain words on every card - no bare identifiers where a judge looks.
function cardTitle(c) {
  const a = c.args || {};
  if (c.tool === "search_flights" && a.destination)
    return `Searching flights · ${cap(a.origin ?? "?")} → ${cap(a.destination)}`;
  if (c.tool === "check_seat_availability" && a.flight_id)
    return `Checking seats · ${a.flight_id}`;
  if (c.tool === "book_flight") {
    const seat = (a.seat || "").split(":")[1] || a.seat || "?";
    return `Booking · seat ${seat} on ${a.flight_id} ×${a.pax ?? 1}`;
  }
  return c.tool;
}

function cardResult(r) {
  if (!r) return "";
  if (r.booking_ref) return `Booked · ref ${r.booking_ref}`;
  if (r.flights) return `Found ${r.flights.length} flights`;
  if (r.seats) return `${r.seats.length} seats free`;
  return r.ok ? "Done" : r.error || "Failed";
}

// Generic beats for scenarios without authored copy.
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

function App() {
  const [scenarios, setScenarios] = useState([]);
  const [sel, setSel] = useState(0);
  const [view, setView] = useState("timeline");
  const [playhead, setPlayhead] = useState(0);
  const [target, setTarget] = useState(null);
  const rawRef = useRef("");
  const threadRef = useRef(null);

  // Data contract unchanged: poll /trace.json once a second, reset only when
  // the file actually changed. Old single-scenario files still work.
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

  const current = scenarios[sel] || { name: "", blurb: "", multimodal: false, trace: [] };
  const trace = current.trace;
  const authored = current.name === FLAGSHIP;

  const model = useMemo(() => {
    const calls = [];
    const byId = {};
    const thread = [];
    const slotChanges = [];
    const lastSeen = {}; // a re-issued call re-states unchanged slots: not a change
    let tEnd = 0;
    for (const e of trace) {
      tEnd = Math.max(tEnd, e.t || 0);
      if (e.kind === "call" && e.dir === "out") {
        const c = {
          id: e.call_id, tool: e.tool, args: e.args, mutating: e.mutating,
          reads: e.reads || [],
          start: e.t, end: null, cancelled: false, cancelT: null, by: [], result: null,
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
        thread.push({ t: e.t, side: "user", text: e.text, correction: e.kind === "interrupt" });
      } else if (e.kind === "frame" || e.kind === "audio") {
        thread.push({ t: e.t, side: "user", text: e.caption || "(audio)", frame: true });
      } else if (e.kind === "say" || e.kind === "final" || e.kind === "clarify") {
        thread.push({ t: e.t, side: "agent", text: e.text, kind: e.kind });
      }
    }
    for (const c of calls) if (c.end === null) c.end = tEnd;
    return { calls, thread, slotChanges, tEnd };
  }, [trace]);

  const beats = useMemo(
    () => (authored ? AUTHORED_BEATS : deriveBeats(trace, model.tEnd)),
    [authored, trace, model.tEnd]
  );

  // Dependency tree: static layout, computed once per scenario. Depth comes
  // from real derivation - a call whose argument value appeared in an earlier
  // call's result sits one row below that call. Edges run slot -> call for
  // every slot in the call's (transitively inherited) reads: that is what
  // the timeline cannot show.
  const tree = useMemo(() => {
    const flat = (v, out = []) => {
      if (v == null) return out;
      if (Array.isArray(v)) v.forEach((x) => flat(x, out));
      else if (typeof v === "object") Object.values(v).forEach((x) => flat(x, out));
      else out.push(String(v));
      return out;
    };
    const calls = model.calls.map((c) => ({ ...c, depth: 1, parent: null, supersededAt: null }));
    for (const c of calls) {
      const argVals = Object.values(c.args || {}).map(String);
      for (const p of calls) {
        if (p === c || !p.result || p.end > c.start + 1e-9) continue;
        if (p.depth >= c.depth && flat(p.result).some((v) => argVals.includes(v))) {
          c.depth = p.depth + 1;
          c.parent = p.id;
        }
      }
      // a completed mutation whose reads went stale afterwards was NOT
      // cancelled - it is still active in the world, and must render so
      if (c.mutating && c.result?.ok && !c.cancelled) {
        const hit = model.slotChanges.find((s) => s.t > c.end && c.reads.includes(s.key));
        if (hit) c.supersededAt = hit.t;
      }
    }
    const maxDepth = Math.max(1, ...calls.map((c) => c.depth));
    const pos = {};
    SLOT_KEYS.forEach((k, i) => {
      pos[k] = { x: ((i + 0.5) / SLOT_KEYS.length) * 100, y: 10 };
    });
    for (let d = 1; d <= maxDepth; d++) {
      const row = calls.filter((c) => c.depth === d).sort((a, b) => a.start - b.start);
      row.forEach((c, j) => {
        pos[c.id] = {
          x: ((j + 0.5) / row.length) * 100,
          y: 10 + (d * 80) / Math.max(maxDepth, 2),
        };
      });
    }
    const edges = [];
    for (const c of calls) {
      for (const s of c.reads) if (pos[s]) edges.push({ from: s, to: c.id, slot: s, call: c });
      if (c.parent) edges.push({ from: c.parent, to: c.id, derived: true, call: c });
    }
    return { calls, edges, pos };
  }, [model]);

  // Stepping: tween the playhead toward the requested beat. Anchored to wall
  // time with a snap timeout so a step always lands even if rAF frames stall.
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

  const beat = beats.reduce((n, b) => (playhead >= b.t - 1e-6 ? n + 1 : n), 0);
  const next = () => beat < beats.length && setTarget(beats[beat].t);
  const back = () => setTarget(beat > 1 ? beats[beat - 2].t : 0);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const seen = (t) => playhead >= t - 1e-9;
  const visibleMsgs = model.thread.filter((m) => seen(m.t));
  const visibleCalls = model.calls.filter((c) => seen(c.start));

  // the thread grows downward; keep the newest message in view
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [visibleMsgs.length]);

  const banner = authored
    ? beat === 0
      ? { n: null, title: cap(current.name), caption: "Step through with Next." }
      : { n: beat, ...AUTHORED_BEATS[beat - 1] }
    : beat === 0
      ? { n: null, title: cap(current.name), caption: "Step through with Next." }
      : { n: beat, title: beats[beat - 1].caption, caption: null };

  return (
    <div className="app">
      <header className="top">
        <span className="brand">PRISM</span>
        <select
          className="picker"
          value={sel}
          onChange={(e) => {
            setSel(Number(e.target.value));
            setPlayhead(0);
            setTarget(null);
          }}
        >
          {scenarios.map((s, i) => (
            <option key={s.name} value={i}>
              {cap(s.name)}{s.multimodal ? "  (multimodal)" : ""}
            </option>
          ))}
        </select>
        <div className="viewtoggle">
          <button className={view === "timeline" ? "on" : ""} onClick={() => setView("timeline")}>
            Timeline
          </button>
          <button className={view === "tree" ? "on" : ""} onClick={() => setView("tree")}>
            Tree
          </button>
        </div>
        <div className="controls">
          <input
            type="range" min="0" max={model.tEnd || 1} step="0.01"
            value={Math.min(playhead, model.tEnd || 1)}
            onChange={(e) => { setTarget(null); setPlayhead(Number(e.target.value)); }}
          />
          <span className="clock mono">t={playhead.toFixed(2)}s</span>
        </div>
      </header>

      {current.blurb && <p className="blurb">{current.blurb}</p>}

      <section className="banner">
        <div className="banner-text">
          <h1>
            {banner.n && <span className="beat-n mono">{banner.n}/{beats.length}</span>}
            {banner.title}
          </h1>
          {banner.caption && <p className="cap">{banner.caption}</p>}
        </div>
        <div className="steps">
          <button className="step back" onClick={back} disabled={beat === 0 && playhead === 0}>
            ← Back
          </button>
          <button className="step next" onClick={next} disabled={beat === beats.length}>
            Next →
          </button>
        </div>
      </section>

      {view === "tree" && (() => {
        // per-slot state at the playhead, shared by nodes and edge anchors
        const slotState = {};
        SLOT_KEYS.forEach((k) => {
          const past = model.slotChanges.filter((s) => s.key === k && seen(s.t));
          slotState[k] = {
            cur: past.length ? past[past.length - 1] : null,
            prev: past.length > 1 ? past[past.length - 2] : null,
          };
        });
        const ANCHOR = 2.4; // % of graph height: stale row above, current below
        return (
        <main className="treewrap">
          <div className="graph">
            <svg className="wires" viewBox="0 0 100 100" preserveAspectRatio="none">
              {tree.edges.map((e, i) => {
                const a = tree.pos[e.from];
                const b = tree.pos[e.to];
                const c = e.call;
                // no wires to calls that don't exist yet: ghost nodes keep
                // the structure visible, but their edges must not compete
                // with the live ones
                if (!a || !b || !seen(c.start)) return null;
                const dead = c.cancelled && seen(c.cancelT);
                // the kill edge burns red while the cancellation is the
                // current subject, then settles: the closing frame is about
                // the booking that resolved, not the call that died
                const HOT = 1.2; // trace-seconds a call stays the subject
                const hot = dead && playhead - c.cancelT < HOT;
                // a call that finished (or whose kill cooled) more than the
                // subject-window ago is old news: its edges stay traceable
                // at ~20% but stop competing with whatever is happening now
                const quiet = dead
                  ? playhead - c.cancelT >= HOT
                  : c.result && seen(c.end) && playhead - c.end >= HOT;
                const kill = !e.derived && dead && c.by.includes(e.slot);
                const cls =
                  "wire" +
                  (e.derived ? " derived" : "") +
                  (kill ? (hot ? " red" : " cooled") : quiet ? " quiet" : "");
                // causal routing: once a slot has changed, the kill edge
                // leaves from the struck stale row, live edges from the
                // current-value row - dead value -> dead call, live -> live
                let y1 = a.y;
                if (!e.derived && slotState[e.slot]?.prev)
                  y1 = a.y + (kill ? -ANCHOR : ANCHOR);
                return <line key={i} className={cls} x1={a.x} y1={y1} x2={b.x} y2={b.y} />;
              })}
            </svg>
            {SLOT_KEYS.map((k) => {
              const { cur, prev } = slotState[k];
              const flash = cur && cur.t > 0 && playhead - cur.t < 0.9 && playhead < model.tEnd;
              const p = tree.pos[k];
              return (
                <div
                  key={k}
                  className={"node slotnode" + (flash ? " flash" : "")}
                  style={{ left: `${p.x}%`, top: `${p.y}%` }}
                >
                  <span className="node-key">{SLOT_NAMES[k]}</span>
                  {prev && (
                    <span className="node-stale mono">
                      <s>{String(prev.value)}</s>
                    </span>
                  )}
                  <span className="node-val mono">{cur === null ? "—" : String(cur.value)}</span>
                </div>
              );
            })}
            {tree.calls.map((c) => {
              const p = tree.pos[c.id];
              const dead = c.cancelled && seen(c.cancelT);
              const stale = c.supersededAt !== null && seen(c.supersededAt);
              const finished = !c.cancelled && seen(c.end) && c.result;
              const running = seen(c.start) && !dead && !finished;
              const cls =
                "node callnode" +
                (seen(c.start) ? "" : " ghost") +
                (dead ? " killed" : stale ? " stale" : finished ? " done" : running ? " running" : "");
              const glyph = dead ? "✕" : stale ? "!" : finished ? "✓" : running ? "▶" : "○";
              return (
                <div key={c.id} className={cls} style={{ left: `${p.x}%`, top: `${p.y}%` }}>
                  <i className="stat mono">{glyph}</i>
                  <span className="node-title">{cardTitle(c)}</span>
                  {dead ? (
                    <span className="node-status red-text">
                      cancelled — {c.by.map((s) => SLOT_NAMES[s] || s).join(", ")} changed
                    </span>
                  ) : finished ? (
                    <span className="node-status">
                      {cardResult(c.result)}
                      {stale && <b className="still mono">STILL ACTIVE</b>}
                    </span>
                  ) : running ? (
                    <span className="node-status">running…</span>
                  ) : (
                    <span className="node-status">not issued yet</span>
                  )}
                </div>
              );
            })}
          </div>
        </main>
        );
      })()}

      {view === "timeline" && (
      <main className="panels">
        {/* ---- conversation thread ---- */}
        <section className="thread" ref={threadRef}>
          {visibleMsgs.map((m, i) => (
            <div key={i} className={"msg " + m.side + (m.kind === "say" ? "" : " strong")}>
              <div
                className={
                  "bubble" +
                  (m.correction ? " correction" : "") +
                  (m.frame ? " frame" : "")
                }
              >
                {m.correction && <span className="tag">correction</span>}
                {m.frame && <span className="tag neutral">camera</span>}
                {m.text}
              </div>
              <span className="stamp mono">{m.t.toFixed(1)}s</span>
            </div>
          ))}
        </section>

        {/* ---- work panel: one card per tool call ---- */}
        <section className="work">
          {visibleCalls.map((c) => {
            const dead = c.cancelled && seen(c.cancelT);
            const finished = !c.cancelled && seen(c.end) && c.result;
            const running = !dead && !finished;
            const dur = Math.max(c.end - c.start, 0.02);
            const fill = Math.max(0, Math.min(playhead, c.end) - c.start) / dur;
            return (
              <div key={c.id} className={"card" + (dead ? " killed" : finished ? " done" : "")}>
                <div className="card-head">
                  <span className="card-title">{cardTitle(c)}</span>
                  <span className="stamp mono">{c.start.toFixed(1)}s</span>
                </div>
                {running && (
                  <div className="progress">
                    <i style={{ width: `${fill * 100}%` }} />
                  </div>
                )}
                {finished && <div className="card-result">{cardResult(c.result)}</div>}
                {dead && (
                  <div className="card-cancel">
                    cancelled — {c.by.map((s) => SLOT_NAMES[s] || s).join(", ")} changed
                  </div>
                )}
              </div>
            );
          })}
        </section>
      </main>
      )}

      {view === "timeline" && (
      <footer className="slots">
        {SLOT_KEYS.map((k) => {
          const past = model.slotChanges.filter((s) => s.key === k && seen(s.t));
          const cur = past.length ? past[past.length - 1] : null;
          const prev = past.length > 1 ? past[past.length - 2] : null;
          const flipped = cur && prev && prev.value !== cur.value;
          const flash = cur && cur.t > 0 && playhead - cur.t < 0.9 && playhead < model.tEnd;
          return (
            <div key={k} className={"slot" + (flash ? " flash" : "")}>
              <span className="slot-key">{SLOT_NAMES[k]}</span>
              <span className="slot-val mono">
                {cur === null ? "—" : flash && flipped ? (
                  <>
                    <s>{String(prev.value)}</s> → {String(cur.value)}
                  </>
                ) : (
                  String(cur.value)
                )}
              </span>
            </div>
          );
        })}
      </footer>
      )}
    </div>
  );
}

export default App;
