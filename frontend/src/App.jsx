import { useState, useEffect, useCallback, useRef } from "react";
import "./App.css";

// Point this at your FastAPI server. Change for production deploys.
const API_BASE = "http://127.0.0.1:8000";

const EMOTION_META = {
  sadness: { emoji: "😢", color: "#5B7FB4", label: "Sadness" },
  joy: { emoji: "😄", color: "#E8A33D", label: "Joy" },
  love: { emoji: "❤️", color: "#C4547A", label: "Love" },
  anger: { emoji: "😠", color: "#C0392B", label: "Anger" },
  fear: { emoji: "😨", color: "#7C5EA3", label: "Fear" },
  surprise: { emoji: "😲", color: "#3FA796", label: "Surprise" },
};

const EMOTION_ORDER = ["sadness", "joy", "love", "anger", "fear", "surprise"];

function StatusDot({ status }) {
  // status: "checking" | "up" | "down"
  const label =
    status === "up" ? "Model ready" : status === "down" ? "Server unreachable" : "Checking…";
  return (
    <div className={`status-pill status-${status}`}>
      <span className="status-dot" />
      <span className="status-text">{label}</span>
    </div>
  );
}

function Spectrum({ probabilities }) {
  const dominant = EMOTION_ORDER.reduce((a, b) =>
    probabilities[b] > probabilities[a] ? b : a
  );

  return (
    <div className="spectrum">
      <div className="spectrum-bar" role="img" aria-label="Emotion probability spectrum">
        {EMOTION_ORDER.map((key) => {
          const pct = probabilities[key] * 100;
          return (
            <div
              key={key}
              className={`spectrum-segment ${key === dominant ? "is-dominant" : ""}`}
              style={{
                width: `${pct}%`,
                background: EMOTION_META[key].color,
              }}
              title={`${EMOTION_META[key].label}: ${pct.toFixed(1)}%`}
            />
          );
        })}
      </div>

      <ul className="spectrum-legend">
        {EMOTION_ORDER.map((key) => (
          <li key={key} className={key === dominant ? "is-dominant" : ""}>
            <span className="legend-swatch" style={{ background: EMOTION_META[key].color }} />
            <span className="legend-label">{EMOTION_META[key].label}</span>
            <span className="legend-value">{(probabilities[key] * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState("checking");
  const pollRef = useRef(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();
      setHealth(data.model_loaded ? "up" : "down");
    } catch {
      setHealth("down");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    pollRef.current = setInterval(checkHealth, 20000);
    return () => clearInterval(pollRef.current);
  }, [checkHealth]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const charCount = text.length;

  return (
    <div className="app">
      <div className="grain" aria-hidden="true" />

      <header className="app-header">
        <div className="brand">
          <span className="brand-eyebrow">Bi‑GRU · 6-way classifier</span>
          <h1>Emotion Reader</h1>
        </div>
        <StatusDot status={health} />
      </header>

      <main className="panel">
        <form onSubmit={handleSubmit} className="input-panel">
          <label htmlFor="text-input" className="panel-label">
            Input
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type a sentence and see what it's carrying…"
            maxLength={2000}
            rows={5}
          />
          <div className="input-footer">
            <span className="char-count">{charCount}/2000</span>
            <button
              type="submit"
              disabled={!text.trim() || loading || health === "down"}
              className="submit-btn"
            >
              {loading ? "Reading…" : "Read emotion"}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}

        {health === "down" && !error && (
          <div className="error-banner">
            Can't reach the model server at {API_BASE}. Start the backend and try again.
          </div>
        )}

        <section className="result-panel" aria-live="polite">
          {result ? (
            <>
              <div className="result-headline">
                <span className="result-emoji">{result.emoji}</span>
                <div>
                  <span className="panel-label">Reading</span>
                  <h2 style={{ color: EMOTION_META[result.predicted_emotion]?.color }}>
                    {EMOTION_META[result.predicted_emotion]?.label ?? result.predicted_emotion}
                  </h2>
                  <p className="confidence">
                    {(result.confidence * 100).toFixed(1)}% confidence
                  </p>
                </div>
              </div>
              <Spectrum probabilities={result.all_probabilities} />
            </>
          ) : (
            <div className="result-empty">
              <span className="panel-label">Reading</span>
              <p>Submit a sentence to see its emotional spectrum.</p>
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>sadness · joy · love · anger · fear · surprise</span>
      </footer>
    </div>
  );
}