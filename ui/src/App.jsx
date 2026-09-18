import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [trace, setTrace] = useState([]);

  useEffect(() => {
    const loadTrace = () => {
      fetch("/trace.json?t=" + Date.now())
        .then((res) => res.json())
        .then((data) => setTrace(data.trace || []))
        .catch((err) => console.error(err));
    };

    loadTrace();

    const interval = setInterval(loadTrace, 1000);

    return () => clearInterval(interval);
  }, []);

  const calls = trace.filter((e) => e.kind === "call");

  const isCancelled = (callId) =>
    trace.some(
      (e) => e.kind === "cancel" && e.call_id === callId
    );

  const getCancelEvent = (callId) =>
    trace.find(
      (e) => e.kind === "cancel" && e.call_id === callId
    );

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>PRISM</h1>
          <p>Agent Trace Visualizer</p>
        </div>

        <div className="live">
          <span></span>
          LIVE TRACE
        </div>
      </header>

      <div className="dashboard">

        {/* TIMELINE */}
        <section className="timeline-card">
          <div className="card-header">
            <div>
              <h2>Execution Timeline</h2>
              <p>Real-time agent events</p>
            </div>

            <span className="event-count">
              {trace.length} EVENTS
            </span>
          </div>

          <div className="timeline">

            {trace.map((event, index) => {
              if (event.kind === "call") {
                const cancelled = isCancelled(event.call_id);
                const cancelEvent = getCancelEvent(event.call_id);

                return (
                  <div className="timeline-row" key={index}>

                    <div className="timestamp">
                      {Number(event.t ?? 0).toFixed(1)}s
                    </div>

                    <div className="line">
                      <div className="timeline-dot"></div>
                    </div>

                    <div className="event-card tool-event">

                      <div className="event-label">
                        <span className="tool-badge">TOOL CALL</span>
                      </div>

                      <div className={`tool-bar ${cancelled ? "cancelled" : ""}`}>
                        <div className="tool-name">
                          {event.tool}
                        </div>

                        <div className="bar">
                          <span></span>
                          {cancelled && <b>╳</b>}
                        </div>

                        <small>{event.call_id}</small>
                      </div>

                      {cancelled && (
                        <div className="cancel-info">
                          <strong>CANCELLED</strong>

                          {cancelEvent?.invalidated_by && (
                            <span>
                              Invalidated by:{" "}
                              {cancelEvent.invalidated_by.join(", ")}
                            </span>
                          )}
                        </div>
                      )}

                    </div>
                  </div>
                );
              }

              return (
                <div className="timeline-row" key={index}>

                  <div className="timestamp">
                    {Number(event.t ?? 0).toFixed(1)}s
                  </div>

                  <div className="line">
                    <div className="timeline-dot"></div>
                  </div>

                  <div className="event-card">

                    <div className="event-label">
                      <span className={`badge ${event.kind}`}>
                        {event.kind}
                      </span>

                      <span className="direction">
                        {event.dir === "in" ? "← IN" : "→ OUT"}
                      </span>
                    </div>

                    <div className="event-text">
                      {event.text || event.tool || ""}
                    </div>

                  </div>
                </div>
              );
            })}

          </div>
        </section>

        {/* SLOTS */}
        <aside className="slot-card">

          <div className="card-header">
            <div>
              <h2>Current Slots</h2>
              <p>Live state</p>
            </div>
          </div>

          <div className="slots">

            <div className="slot">
              <span>destination</span>
              <strong>Goa</strong>
            </div>

            <div className="slot">
              <span>origin</span>
              <strong>Delhi</strong>
            </div>

            <div className="slot">
              <span>date</span>
              <strong>—</strong>
            </div>

            <div className="slot">
              <span>passengers</span>
              <strong>—</strong>
            </div>

          </div>

          <div className="legend">
            <div>
              <span className="legend-dot"></span>
              Event
            </div>

            <div>
              <span className="legend-x">╳</span>
              Cancelled
            </div>
          </div>

        </aside>

      </div>
    </div>
  );
}

export default App;
