import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// One screen, one story: a tool call dies at the instant a slot changes,
// while unrelated calls keep running. Everything shares a single time axis
// so "while" is visible without scrolling.

const SLOT_KEYS = ["origin", "destination", "date", "pax"];
const SPEED = 2; // trace-seconds per wall-second (~2x real time)
const LOOP_PAUSE_MS = 1400; // hold on the finished frame, then loop

function summarise(result) {
  if (!result) return "";
  if (result.booking_ref) return result.booking_ref;
  if (result.flights) return `${result.flights.length} flights`;
  if (result.seats) return `${result.seats.length} seats`;
  return result.ok ? "ok" : result.error || "failed";
}

function argHint(args) {
  if (!args) return "";
  if (args.seat) return `${args.seat} ×${args.pax ?? 1}`;
  if (args.flight_id) return args.flight_id;
  if (args.destination) return `${args.origin ?? "?"}→${args.destination}`;
  return "";
}

function App() {
  const [trace, setTrace] = useState([]);
  const [scenario, setScenario] = useState("");
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(true);
  const rawRef = useRef("");
  const clockRef = useRef({ last: 0, holdUntil: 0 });
  const liveRef = useRef({ playing: true, tEnd: 0 });

  // Data contract unchanged: poll /trace.json once a second. Only reset the
  // animation when the file actually changed.
  useEffect(() => {
    const loadTrace = () => {
      fetch("/trace.json?t=" + Date.now())
        .then((res) => res.json())
        .then((data) => {
          const s = JSON.stringify(data.trace || []);
          if (s !== rawRef.current) {
            rawRef.current = s;
            setTrace(data.trace || []);
            setScenario(data.scenario || "");
            setPlayhead(0);
          }
        })
        .catch((err) => console.error(err));
    };
    loadTrace();
    const interval = setInterval(loadTrace, 1000);
    return () => clearInterval(interval);
  }, []);

  const model = useMemo(() => {
    const calls = [];
    const byId = {};
    const utterances = [];
    const speech = [];
    const slotChanges = []; // {t, key, value} recovered from call args
    const lastSeen = {}; // a re-issued call re-states unchanged slots: not a change
    let tEnd = 0;
    for (const e of trace) {
      tEnd = Math.max(tEnd, e.t || 0);
      if (e.kind === "call" && e.dir === "out") {
        const lane = {
          id: e.call_id, tool: e.tool, args: e.args, start: e.t,
          end: null, cancelled: false, cancelT: null, by: [], result: null,
        };
        calls.push(lane);
        byId[e.call_id] = lane;
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
        utterances.push(e);
      } else if (e.kind === "say" || e.kind === "final") {
        speech.push(e);
      }
    }
    for (const c of calls) if (c.end === null) c.end = tEnd;
    const T = Math.max(tEnd, 0.001) * 1.08; // breathing room on the right
    const cancels = calls.filter((c) => c.cancelled);
    return { calls, utterances, speech, slotChanges, cancels, T, tEnd };
  }, [trace]);

  liveRef.current = { playing, tEnd: model.tEnd };

  // Autoplay: rAF drives the playhead; bars grow because their fill width is
  // a pure function of the playhead. Loop with a short hold at the end.
  useEffect(() => {
    let raf;
    const step = (now) => {
      const c = clockRef.current;
      const dt = c.last ? (now - c.last) / 1000 : 0;
      c.last = now;
      const { playing, tEnd } = liveRef.current;
      if (playing && tEnd > 0 && now >= c.holdUntil) {
        setPlayhead((p) => {
          if (p >= tEnd) return 0; // hold finished: loop
          const n = p + dt * SPEED;
          if (n >= tEnd) {
            c.holdUntil = now + LOOP_PAUSE_MS;
            return tEnd;
          }
          return n;
        });
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  const x = (t) => `${(t / model.T) * 100}%`;
  const w = (d) => `${(d / model.T) * 100}%`;
  const seen = (t) => playhead >= t - 1e-9;

  const scrub = (v) => {
    clockRef.current.holdUntil = 0;
    setPlayhead(Number(v));
  };

  const gridSeconds = [];
  for (let s = 0; s <= Math.floor(model.tEnd); s++) gridSeconds.push(s);

  return (
    <div className="app">
      <header className="top">
        <div className="title">
          <span className="brand">PRISM</span>
          <span className="scenario">{scenario}</span>
        </div>
        <div className="legend">
          <span><i className="sw running" /> running</span>
          <span><i className="sw done" /> completed</span>
          <span><i className="sw killed" /> cancelled</span>
        </div>
        <div className="controls">
          <button className="playbtn" onClick={() => setPlaying((p) => !p)}>
            {playing ? "❚❚" : "▶"}
          </button>
          <input
            type="range" min="0" max={model.tEnd || 1} step="0.01"
            value={Math.min(playhead, model.tEnd || 1)}
            onChange={(e) => scrub(e.target.value)}
          />
          <span className="clock mono">t={playhead.toFixed(2)}s</span>
        </div>
      </header>

      <main className="board">
        {/* ---- utterance track ---- */}
        <div className="row utt-row">
          <div className="label">user</div>
          <div className="rail">
            {model.utterances.map((u, i) => (
              <div
                key={i}
                className={
                  "utt" +
                  (u.kind === "interrupt" ? " correction" : "") +
                  (seen(u.t) ? "" : " future") +
                  (i % 2 ? " low" : "") +
                  (u.t / model.T > 0.72 ? " end" : "")
                }
                style={{ left: x(u.t) }}
              >
                <span className="utt-text">
                  “{u.text}”<b className="mono"> {u.t.toFixed(1)}s</b>
                </span>
                <span className="utt-dot" />
              </div>
            ))}
          </div>
        </div>

        {/* ---- call lanes ---- */}
        {model.calls.map((c) => {
          const started = seen(c.start);
          const dur = Math.max(c.end - c.start, 0.02);
          const fill = Math.max(0, Math.min(playhead, c.end) - c.start) / dur;
          const dead = c.cancelled && seen(c.cancelT);
          const finished = !c.cancelled && seen(c.end) && c.result;
          const cls =
            "bar" +
            (dead ? " killed" : finished ? " done" : started ? " running" : "");
          return (
            <div className="row lane" key={c.id}>
              <div className={"label" + (started ? "" : " future")}>
                <span className="mono tool">{c.tool}</span>
                <span className="mono meta">{c.id} · {argHint(c.args)}</span>
              </div>
              <div className="rail">
                <div className={cls} style={{ left: x(c.start), width: w(dur) }}>
                  <i className="fill" style={{ width: `${fill * 100}%` }} />
                  <i className="strike" />
                </div>
                {c.cancelled && (
                  <div
                    className={"invalidated mono" + (dead ? " show" : "")}
                    style={{ left: x(c.start) }}
                  >
                    invalidated_by: {c.by.join(", ")}
                  </div>
                )}
                {finished && (
                  <div className="result mono" style={{ left: x(c.end) }}>
                    {summarise(c.result)}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* ---- agent speech strip ---- */}
        <div className="row speech-row">
          <div className="label">agent</div>
          <div className="rail">
            {model.speech.map((s, i) => (
              <div
                key={i}
                className={
                  "tick" +
                  (s.kind === "final" ? " final" : "") +
                  (seen(s.t) ? "" : " future") +
                  (i % 2 ? " low" : "") +
                  (s.t / model.T > 0.72 ? " end" : "")
                }
                style={{ left: x(s.t) }}
              >
                <span className="tick-mark" />
                <span className="tick-text">{s.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ---- time axis ---- */}
        <div className="row axis-row">
          <div className="label" />
          <div className="rail">
            {gridSeconds.map((s) => (
              <span key={s} className="axis-label mono" style={{ left: x(s) }}>
                {s}s
              </span>
            ))}
          </div>
        </div>

        {/* ---- overlay: gridlines, hero cancel line, playhead ---- */}
        <div className="overlay">
          {gridSeconds.map((s) => (
            <i key={s} className="grid" style={{ left: x(s) }} />
          ))}
          {model.cancels.map((c) => (
            <i
              key={c.id}
              className={"heroline" + (seen(c.cancelT) ? " show" : "")}
              style={{ left: x(c.cancelT) }}
            />
          ))}
          <i className="playhead" style={{ left: x(Math.min(playhead, model.tEnd)) }} />
        </div>
      </main>

      {/* ---- slot panel: state at the playhead, not final state ---- */}
      <footer className="slots">
        {SLOT_KEYS.map((k) => {
          const past = model.slotChanges.filter((s) => s.key === k && seen(s.t));
          const cur = past.length ? past[past.length - 1] : null;
          const prev = past.length > 1 ? past[past.length - 2] : null;
          const flipped = cur && prev && prev.value !== cur.value;
          const flash = cur && cur.t > 0 && playhead - cur.t < 0.9 && playhead < model.tEnd;
          return (
            <div key={k} className={"slot" + (flash ? " flash" : "")}>
              <span className="slot-key">{k}</span>
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
    </div>
  );
}

export default App;
