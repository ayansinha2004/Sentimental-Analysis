import { useState, useEffect, useRef } from "react";
import "./App.css";

// ✅ New: Automatically strips trailing slashes and ensures https:// protocol
const rawApiUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const API_BASE = rawApiUrl.startsWith("http") 
  ? rawApiUrl.replace(/\/+$/, "") 
  : `https://${rawApiUrl.replace(/\/+$/, "")}`;
const EMOTION_META = {
  sadness: { emoji: "😢", color: "var(--sadness)", label: "Sadness" },
  joy: { emoji: "😄", color: "var(--joy)", label: "Joy" },
  love: { emoji: "❤️", color: "var(--love)", label: "Love" },
  anger: { emoji: "😠", color: "var(--anger)", label: "Anger" },
  fear: { emoji: "😨", color: "var(--fear)", label: "Fear" },
  surprise: { emoji: "😲", color: "var(--surprise)", label: "Surprise" },
};

function StatusPill() {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setStatus(data.model_loaded ? "ready" : "loading");
      })
      .catch(() => {
        if (!cancelled) setStatus("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const labels = {
    checking: "checking",
    ready: "model ready",
    loading: "model loading",
    offline: "server offline",
  };

  return (
    <div className={`status-pill status-${status}`}>
      <span className="status-dot" />
      {labels[status]}
    </div>
  );
}

function ResultBars({ probabilities, topEmotion }) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="bars">
      {sorted.map(([emotion, prob], i) => {
        const meta = EMOTION_META[emotion];
        const isTop = emotion === topEmotion;
        return (
          <div className="bar-row" key={emotion}>
            <span className="bar-label">
              {meta.emoji} {meta.label}
            </span>
            <div className="bar-track">
              <div
                className={`bar-fill ${isTop ? "bar-fill-top" : ""}`}
                style={{
                  width: `${prob * 100}%`,
                  background: meta.color,
                  transitionDelay: `${i * 60}ms`,
                }}
              />
            </div>
            <span className="bar-value">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const panelRef = useRef(null);

  const accentColor = result
    ? EMOTION_META[result.predicted_emotion]?.color
    : null;

  const handleAnalyze = async () => {
    if (!text.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (res.status === 503) {
        throw new Error(
          "The model hasn't finished loading yet. Wait a moment and try again."
        );
      }
      if (!res.ok) {
        throw new Error("The analyzer rejected that request. Try a shorter sentence.");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          "Couldn't reach the analyzer. Is the FastAPI server running on port 8000?"
        );
      } else {
        setError(err.message);
      }
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleAnalyze();
    }
  };

  return (
    <div
      className="app"
      style={accentColor ? { "--accent": accentColor } : undefined}
    >
      <header className="header">
        <span className="eyebrow">emotion analyzer</span>
        <StatusPill />
      </header>

      <main className="main">
        <h1 className="headline">
          How are you <em>really</em> feeling?
        </h1>
        <p className="subtext">
          Type a sentence. A bidirectional GRU trained on six emotions reads
          it back — case-folded, stripped of punctuation, run through the
          same pipeline it was trained on.
        </p>

        <div className="console">
          <textarea
            className="input"
            placeholder="I can't believe how well today went..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            maxLength={2000}
            rows={4}
          />
          <div className="console-footer">
            <span className="char-count">{text.length} / 2000</span>
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={!text.trim() || isLoading}
            >
              {isLoading ? "Analyzing…" : "Analyze →"}
            </button>
          </div>
        </div>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <section className="results" ref={panelRef}>
            <div className="top-result">
              <span className="top-emoji">
                {EMOTION_META[result.predicted_emotion]?.emoji}
              </span>
              <div>
                <div className="top-label">
                  {EMOTION_META[result.predicted_emotion]?.label}
                </div>
                <div className="top-confidence">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>
            </div>

            <ResultBars
              probabilities={result.all_probabilities}
              topEmotion={result.predicted_emotion}
            />
          </section>
        )}
      </main>

      <footer className="footer">BiGRU · FastAPI · six-way emotion classification</footer>
    </div>
  );
}
