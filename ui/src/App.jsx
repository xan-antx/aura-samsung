import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// Two modes. Replay: the stepped walkthrough over exported scenario traces
// (unchanged). Live: a chat driving the real agent over a WebSocket - same
// Timeline and Tree, fed from the live stream, playhead following real time.

const SLOT_KEYS = ["origin", "destination", "date", "pax"];
const SLOT_NAMES = { origin: "origin", destination: "destination", date: "date", pax: "passengers" };
const STEP_SPEED = 3.5;
const FLAGSHIP = "mid-utterance destination change";
const PARTIAL_PAUSE_MS = 600;   // typing pause before a non-final chunk is sent

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
  if (c.tool === "cancel_booking")
    return `Cancelling booking ${a.booking_ref}`;
  return c.tool;
}

function cardResult(r) {
  if (!r) return "";
  if (r.booking_ref) return `Booked · ref ${r.booking_ref}`;
  if (r.cancelled) return `Cancelled · ref ${r.cancelled}`;
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

function App() {
  const [mode, setMode] = useState("replay");
  const [scenarios, setScenarios] = useState([]);
  const [sel, setSel] = useState(0);
  const [view, setView] = useState("timeline");
  const [playhead, setPlayhead] = useState(0);
  const [target, setTarget] = useState(null);
  const rawRef = useRef("");
  const threadRef = useRef(null);

  // ---- live mode state ----
  const [liveTrace, setLiveTrace] = useState([]);
  const [liveStatus, setLiveStatus] = useState({ llm: null, conn: "off", latencies: {} });
  const [livePh, setLivePh] = useState(0);
  const [draft, setDraft] = useState("");
  const wsRef = useRef(null);
  const anchorRef = useRef({ sn: 0, pf: 0 });
  const lastSentRef = useRef("");
  const draftTimerRef = useRef(null);

  // Data contract unchanged: poll /trace.json once a second (replay data).
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

  // ---- live websocket ----
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

  // live playhead follows the real clock, anchored to server time. Interval-
  // driven, not rAF: a backgrounded or occluded tab still gets timer ticks,
  // so the live view never freezes; rAF only adds smoothness when available.
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
    return { calls, thread, slotChanges, tEnd };
  }, [trace]);

  const beats = useMemo(
    () => (mode === "live" ? [] : authored ? AUTHORED_BEATS : deriveBeats(trace, model.tEnd)),
    [mode, authored, trace, model.tEnd]
  );

  // Dependency tree: static layout, computed once per trace.
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

  // Replay stepping: tween the playhead toward the requested beat.
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

  const T = mode === "live" ? Math.max(model.tEnd, ph, 8) * 1.08 : Math.max(model.tEnd, 0.001) * 1.08;
  const seen = (t) => ph >= t - 1e-9;

  // ---- live chat input: act before Enter ----
  const sendChunk = (text, final) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1 && text) ws.send(JSON.stringify({ type: "chunk", text, final }));
  };
  const onDraft = (v) => {
    setDraft(v);
    clearTimeout(draftTimerRef.current);
    draftTimerRef.current = setTimeout(() => {
      // a pause at a word boundary: send the words completed so far,
      // cumulative, as a NON-final chunk - never a half-typed word
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

  // thread display: in live mode, superseded partials collapse away and the
  // agent acting between a partial and its Enter is marked visibly
  const actsCC = useMemo(
    () => trace.filter((e) => e.dir === "out" && (e.kind === "call" || e.kind === "cancel")),
    [trace]
  );
  const userEvs = useMemo(
    () => model.thread.filter((m) => m.side === "user"),
    [model]
  );
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
      // strictly before this final: actions born in the final's own dispatch
      // share its timestamp and do not count as acting early
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
      : { n: beat, title: beats[beat - 1].caption, caption: null };

  return (
    <div className="app">
      <header className="top">
        <span className="brand">PRISM</span>
        <div className="viewtoggle">
          <button className={mode === "replay" ? "on" : ""} onClick={() => setMode("replay")}>
            Replay
          </button>
          <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>
            Live
          </button>
        </div>
        {mode === "replay" && (
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
        )}
        <div className="viewtoggle">
          <button className={view === "timeline" ? "on" : ""} onClick={() => setView("timeline")}>
            Timeline
          </button>
          <button className={view === "tree" ? "on" : ""} onClick={() => setView("tree")}>
            Tree
          </button>
        </div>
        <div className="controls">
          {mode === "replay" && (
            <input
              type="range" min="0" max={model.tEnd || 1} step="0.01"
              value={Math.min(playhead, model.tEnd || 1)}
              onChange={(e) => { setTarget(null); setPlayhead(Number(e.target.value)); }}
            />
          )}
          <span className="clock mono">t={ph.toFixed(2)}s</span>
        </div>
      </header>

      {mode === "replay" && current.blurb && <p className="blurb">{current.blurb}</p>}

      {mode === "replay" && (
        <section className="banner">
          <div className="banner-text">
            <h1 className={authored ? "" : "h-blurb"}>
              {banner.n && <span className="beat-n mono">{banner.n}/{beats.length}</span>}
              {authored ? banner.title : (current.blurb && beat > 0 ? current.blurb : banner.title)}
            </h1>
            {(authored ? banner.caption : beat > 0 ? beats[beat - 1].caption : banner.caption) && (
              <p className="cap">
                {authored ? banner.caption : beat > 0 ? beats[beat - 1].caption : banner.caption}
              </p>
            )}
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
      )}

      {mode === "live" && (
        <section className="livebar">
          <span className={"conn " + liveStatus.conn}>
            {liveStatus.conn === "live" ? "● connected" :
             liveStatus.conn === "connecting" ? "○ connecting…" : "○ disconnected"}
          </span>
          <span className={"llmflag mono" + (liveStatus.state === "failing" ? " red-text" : "")}>
            extraction: {
              liveStatus.state === "ok" ? "LLM (live)"
              : liveStatus.state === "failing" ? "LLM FAILING — deterministic fallback"
              : liveStatus.state === "untried" ? "LLM configured — no call yet"
              : liveStatus.state === "off" ? "deterministic (no API key)"
              : "…"}
          </span>
          <span className="hint">pause mid-sentence and the agent acts before you press Enter</span>
          <button className="step" onClick={doReset}>Reset</button>
        </section>
      )}

      {view === "tree" && (() => {
        const slotState = {};
        SLOT_KEYS.forEach((k) => {
          const past = model.slotChanges.filter((s) => s.key === k && seen(s.t));
          slotState[k] = {
            cur: past.length ? past[past.length - 1] : null,
            prev: past.length > 1 ? past[past.length - 2] : null,
          };
        });
        const ANCHOR = 2.4;
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
                const cls =
                  "wire" +
                  (e.derived ? " derived" : "") +
                  (kill ? (hot ? " red" : " cooled") : quiet ? " quiet" : "");
                let y1 = a.y;
                if (!e.derived && slotState[e.slot]?.prev)
                  y1 = a.y + (kill ? -ANCHOR : ANCHOR);
                return <line key={i} className={cls} x1={a.x} y1={y1} x2={b.x} y2={b.y} />;
              })}
            </svg>
            {SLOT_KEYS.map((k) => {
              const { cur, prev } = slotState[k];
              const flash = cur && cur.t > 0 && ph - cur.t < 0.9 && (mode === "live" || ph < model.tEnd);
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
        <section className="threadwrap">
          <div className="thread" ref={threadRef}>
            {displayThread.filter((m) => seen(m.t)).map((m, i) => (
              <div
                key={i}
                className={
                  "msg " + m.side +
                  (m.kind === "say" ? "" : " strong") +
                  (mode === "live" && m.partial ? " partial" : "")
                }
              >
                <div
                  className={
                    "bubble" +
                    (m.correction ? " correction" : "") +
                    (m.frame ? " frame" : "")
                  }
                >
                  {m.correction && <span className="tag">correction</span>}
                  {m.frame && <span className="tag neutral">camera</span>}
                  {mode === "live" && m.partial && <span className="tag neutral">typing…</span>}
                  {actedEarly(m) && <span className="tag early">⚡ acted before Enter</span>}
                  {m.text}
                </div>
                <span className="stamp mono">{m.t.toFixed(1)}s</span>
              </div>
            ))}
          </div>
          {mode === "live" && (
            <div className="chatline">
              <input
                className="chatinput"
                placeholder={liveStatus.conn === "live" ? "Talk to the agent…" : "not connected"}
                value={draft}
                disabled={liveStatus.conn !== "live"}
                onChange={(e) => onDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && onEnter()}
              />
              <button className="step next" onClick={onEnter}>Send</button>
            </div>
          )}
        </section>

        <section className="work">
          {model.calls.filter((c) => seen(c.start)).map((c) => {
            const dead = c.cancelled && seen(c.cancelT);
            const finished = !c.cancelled && seen(c.end) && c.result;
            const running = !dead && !finished;
            // live: an in-flight call has no end time yet - progress runs
            // against the server-declared latency for that tool
            const dur = Math.max(c.end - c.start, 0.02);
            const fill = mode === "live" && running && !c.result
              ? Math.min((ph - c.start) / (liveStatus.latencies[c.tool] || 3), 1)
              : Math.max(0, Math.min(ph, c.end) - c.start) / dur;
            return (
              <div key={c.id} className={"card" + (dead ? " killed" : finished ? " done" : "")}>
                <div className="card-head">
                  <span className="card-title">{cardTitle(c)}</span>
                  <span className="stamp mono">{c.start.toFixed(1)}s</span>
                </div>
                {running && (
                  <div className="progress">
                    <i style={{ width: `${Math.min(fill, 1) * 100}%` }} />
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
          const flash = cur && cur.t > 0 && ph - cur.t < 0.9 && (mode === "live" || ph < model.tEnd);
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
