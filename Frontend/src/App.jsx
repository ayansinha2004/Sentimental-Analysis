import { useState, useEffect, useRef } from "react";
import "./App.css";

// 1. Robust URL Normalization (Strips Markdown & double http prefixes)
let rawUrl = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").trim();
rawUrl = rawUrl.replace(/\[|\]|\(|\)/g, ""); // Remove stray markdown brackets
if (rawUrl.includes("http")) {
  const matches = rawUrl.match(/https?:\/\/[^\s]+/g);
  if (matches) rawUrl = matches[0];
}
const API_BASE = rawUrl.startsWith("http")
  ? rawUrl.replace(/\/+$/, "")
  : `https://${rawUrl.replace(/^\/+|\/+$/g, "")}`;

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
    let timerId = null;

    const checkHealth = async () => {
      console.log("[StatusPill] Fetching:", `${API_BASE}/health`);
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
        const data = await res.json();
        if (!cancelled) setStatus(data.model_loaded ? "ready" : "loading");
      } catch (err) {
        console.error("[StatusPill] Failed:", err);
        if (!cancelled) {
          setStatus("offline");
          timerId = setTimeout(checkHealth, 5000); 
        }
      }
    };

    checkHealth();

    return () => {
      cancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  const labels = {
    checking: "checking...",
    ready: "model ready",
    loading: "model loading...",
    offline: "waking server...",
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
        const meta = EMOTION_META[emotion] || { emoji: "❓", color: "#888", label: emotion };
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

  const accentColor = result ? EMOTION_META[result.predicted_emotion]?.color : null;

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
        throw new Error("Model is still loading on the server. Please wait...");
      }
      if (!res.ok) {
        throw new Error("Request failed. Try a shorter sentence.");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof TypeError ? "Network error. Check connection or ad blocker." : err.message);
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
    <div className="app" style={accentColor ? { "--accent": accentColor } : undefined}>
      <header className="header">
        <span className="eyebrow">emotion analyzer</span>
        <StatusPill />
      </header>

      <main className="main">
        <h1 className="headline">
          How are you <em>really</em> feeling?
        </h1>
        <p className="subtext">
          Type a sentence. A bidirectional GRU trained on six emotions reads it back.
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