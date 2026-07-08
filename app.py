import os, json, sys, traceback, tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ---------------------------------------------------------------------------
# CUSUM state classification
#
# Four states, driven by the upper/lower CUSUM alert flags produced by
# CUSUMDetector.update_score() (see cusum.py):
#   1 = stable        -> upper below h AND lower below h
#   2 = upper_alert    -> upper crossed h, lower did not (drifting above baseline)
#   3 = lower_alert    -> lower crossed h, upper did not (drifting below baseline)
#   4 = both_alert     -> both crossed h (oscillation)
# ---------------------------------------------------------------------------

CUSUM_STATUS_TEXT = {
    1: {
        "state": "stable",
        "title": "Within your normal range",
        "message": "Everything looks normal. Your current patterns are staying within your usual range. Keep maintaining your healthy routine.",
    },
    2: {
        "state": "upper_alert",
        "title": "Drifting above baseline",
        "message": "We've noticed your readings are moving above your normal range. This may indicate your well-being is changing. Consider taking a moment to rest and monitor your condition.",
    },
    3: {
        "state": "lower_alert",
        "title": "Drifting below baseline",
        "message": "Your readings are lower than your usual range. This could mean your body is calming down or responding differently than usual. Continue monitoring to ensure everything remains on track.",
    },
    4: {
        "state": "both_alert",
        "title": "Unusual oscillation detected",
        "message": "Your readings are fluctuating above and below your normal range. This unusual pattern may require closer attention. Keep monitoring your condition and consider seeking guidance if it continues.",
    },
}


def classify_cusum_state(alert_upper: bool, alert_lower: bool) -> int:
    """Map a pair of CUSUM alert flags to one of the four states above."""
    if alert_upper and alert_lower:
        return 4
    if alert_upper:
        return 2
    if alert_lower:
        return 3
    return 1


def build_cusum_status(cusum_alert_upper: list, cusum_alert_lower: list, timestamps: list = None) -> dict:
    states = [
        classify_cusum_state(u, l)
        for u, l in zip(cusum_alert_upper, cusum_alert_lower)
    ]
    latest_code = states[-1] if states else 1
    latest = CUSUM_STATUS_TEXT[latest_code]

    alert_indices = [i for i, s in enumerate(states) if s != 1]
    had_alert_history = len(alert_indices) > 0
    last_alert_index = alert_indices[-1] if had_alert_history else None
    last_alert_code = states[last_alert_index] if had_alert_history else None
    last_alert_date = (
        timestamps[last_alert_index]
        if had_alert_history and timestamps and last_alert_index < len(timestamps)
        else None
    )

    result = {
        "states": states,
        "current_code": latest_code,
        "current_state": latest["state"],
        "current_title": latest["title"],
        "current_message": latest["message"],
        "had_alert_history": had_alert_history,
    }

    if had_alert_history and latest_code == 1:
        last_alert_state = CUSUM_STATUS_TEXT[last_alert_code]["state"].replace("_", " ")
        result["current_state"] = "recovered"
        result["current_title"] = "Back within your normal range"
        date_part = f" around {last_alert_date}" if last_alert_date else ""
        result["current_message"] = (
            f"You're currently within your usual range. There was a notable {last_alert_state}{date_part} "
            "before things returned to baseline — worth keeping in mind alongside the current stability."
        )
        result["last_alert_date"] = last_alert_date
        result["last_alert_state"] = last_alert_state

    return result

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mental Health Digital Twin</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0a0a;
    --surface:  #111111;
    --border:   #1f1f1f;
    --muted:    #3a3a3a;
    --text:     #d4d4d4;
    --dim:      #6b6b6b;
    --accent:   #e2e2e2;
    --blue:     #4a90d9;
    --red:      #c0392b;
    --green:    #27ae60;
    --amber:    #d4a017;
    --purple:   #7b68ee;
  }

  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }

  header {
    padding: 32px 48px 24px;
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    font-size: 18px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: -0.01em;
  }
  header p {
    font-size: 12px;
    color: var(--dim);
    margin-top: 4px;
  }

  .container { max-width: 960px; margin: 0 auto; padding: 40px 24px; }

  .section { margin-bottom: 48px; }
  .section-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
  }

  .upload-area {
    border: 1px dashed var(--muted);
    border-radius: 6px;
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s;
  }
  .upload-area:hover { border-color: var(--dim); }
  .upload-area p { color: var(--dim); font-size: 13px; }
  .upload-area strong { color: var(--text); }
  input[type=file] { display: none; }

  .field {
    margin-top: 16px;
  }
  .field label { display: block; font-size: 12px; color: var(--dim); margin-bottom: 6px; }
  .field input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 9px 12px;
    color: var(--text);
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color .15s;
  }
  .field input:focus { border-color: var(--muted); }

  .actions { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  .btn {
    padding: 8px 20px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
    color: var(--text);
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    transition: background .15s, border-color .15s;
  }
  .btn:hover { background: var(--border); border-color: var(--muted); }
  .btn-primary { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 500; }
  .btn-primary:hover { background: #c8c8c8; border-color: #c8c8c8; }

  #status {
    margin-top: 14px;
    font-size: 12px;
    color: var(--dim);
    min-height: 18px;
  }
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 1.5px solid var(--muted);
    border-top-color: var(--text);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #results { display: none; }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 48px 0;
  }

  /* Risk block */
  .risk-block {
    display: flex;
    align-items: flex-start;
    gap: 40px;
    flex-wrap: wrap;
    padding: 28px 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }
  .risk-main .level {
    font-size: 36px;
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .risk-main .sublabel { font-size: 11px; color: var(--dim); margin-top: 6px; }
  .risk-stats { display: flex; gap: 32px; flex-wrap: wrap; }
  .stat { }
  .stat .val { font-size: 22px; font-weight: 300; letter-spacing: -0.02em; }
  .stat .lbl { font-size: 11px; color: var(--dim); margin-top: 3px; }

  .col-low      { color: var(--green); }
  .col-moderate { color: var(--amber); }
  .col-high     { color: var(--red); }
  .col-blue     { color: var(--blue); }
  .col-purple   { color: var(--purple); }
  .col-dim      { color: var(--dim); }

  /* Charts */
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
  .chart-wrap canvas { max-height: 200px; }
  .chart-label { font-size: 12px; color: var(--dim); margin-bottom: 10px; }

  /* Emotions */
  .emotion-list { display: flex; flex-wrap: wrap; gap: 8px; }
  .em-tag {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid var(--border);
    color: var(--dim);
  }
  .em-neg { border-color: #3a1a1a; color: #c0392b; }
  .em-pos { border-color: #1a2e1a; color: #27ae60; }

  /* Intervention note */
  .note {
    font-size: 12px;
    color: var(--dim);
    border-left: 2px solid var(--muted);
    padding-left: 12px;
    margin-top: 20px;
    line-height: 1.7;
  }
  .note.warn { border-color: var(--red); color: #a0522d; }
  .note.ok   { border-color: var(--green); color: #2e7d52; }

  /* CUSUM status banner */
  .cusum-banner {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 18px 20px;
    border-radius: 6px;
    border: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .cusum-banner .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
  }
  .cusum-banner .body .title { font-size: 13px; font-weight: 500; color: var(--accent); margin-bottom: 4px; }
  .cusum-banner .body .msg   { font-size: 12.5px; color: var(--dim); line-height: 1.7; }
  .cusum-stable  { border-color: #1a2e1a; } .cusum-stable  .dot { background: var(--green); }
  .cusum-upper   { border-color: #3a1a1a; } .cusum-upper   .dot { background: var(--red); }
  .cusum-lower   { border-color: #16283a; } .cusum-lower   .dot { background: var(--blue); }
  .cusum-both    { border-color: #2e2a1a; } .cusum-both    .dot { background: var(--purple); }
  .cusum-recovered { border-color: #2e2a1a; } .cusum-recovered .dot { background: var(--amber); }

  .progress-bar {
    width: 100%;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 10px;
  }
  .progress-fill {
    height: 100%;
    background: var(--blue);
    transition: width .3s;
  }

  @media (max-width: 600px) {
    .chart-grid { grid-template-columns: 1fr; }
    .risk-block { flex-direction: column; gap: 24px; }
    header { padding: 24px 20px 18px; }
  }
</style>
</head>
<body>

<header>
  <h1>Mental Health Digital Twin</h1>
  <p>A quiet look at how you've been doing lately, based on what you've written.</p>
</header>

<div class="container">

  <div class="section" id="uploadSection">
    <div class="section-label">Share your entries</div>
    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <p id="dropLabel"><strong>Click to upload</strong> &nbsp;or drag and drop</p>
      <p style="margin-top:6px;font-size:11px;">CSV &middot; JSON &middot; TXT &middot; PDF &middot; DOCX</p>
    </div>
    <input type="file" id="fileInput" accept=".csv,.json,.txt,.pdf,.docx,.doc">

    <div class="field">
      <label>User ID</label>
      <input type="text" id="userId" value="rohith_ms" placeholder="e.g. patient_001">
    </div>

    <div class="actions">
      <button class="btn btn-primary" onclick="submitFile()">Run analysis</button>
      <button class="btn" onclick="runDemo(false)">Demo &mdash; healthy arc</button>
      <button class="btn" onclick="runDemo(true)">Demo &mdash; declining arc</button>
    </div>

    <div id="status"></div>
  </div>

  <div id="results">
    <hr class="divider">

    <div class="section">
      <div class="section-label">How you're doing</div>
      <div class="risk-block">
        <div class="risk-main">
          <div class="level" id="riskLevel">—</div>
          <div class="sublabel">overall picture</div>
        </div>
        <div class="risk-stats" id="riskStats"></div>
      </div>
      <div class="note" id="interventionNote"></div>
    </div>

    <hr class="divider">

    <div class="section">
      <div class="section-label">Mood and risk over the last two weeks</div>
      <div class="chart-grid">
        <div class="chart-wrap">
          <div class="chart-label">How your tone has shifted day to day</div>
          <canvas id="sentimentChart"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label">Moments that stood out as unusual</div>
          <canvas id="anomalyChart"></canvas>
        </div>
      </div>
    </div>

    <hr class="divider">

    <div class="section">
      <div class="section-label">Your personal baseline</div>
      <div class="cusum-banner" id="baselineBanner">
        <div class="dot"></div>
        <div class="body">
          <div class="title" id="baselineTitle">—</div>
          <div class="msg" id="baselineMessage"></div>
        </div>
      </div>
      <div class="chart-label" id="calibrationLabel" style="margin-bottom:6px;"></div>
      <div class="progress-bar"><div class="progress-fill" id="calibrationFill" style="width:0%;"></div></div>
    </div>

    <hr class="divider">

    <div class="section">
      <div class="section-label">Trend stability (CUSUM)</div>
      <div class="cusum-banner" id="cusumBanner">
        <div class="dot"></div>
        <div class="body">
          <div class="title" id="cusumTitle">—</div>
          <div class="msg" id="cusumMessage"></div>
        </div>
      </div>
      <div class="chart-label" style="margin-bottom:14px;">
        Cumulative drift above (upper) and below (lower) your baseline. A dashed line marks the alert threshold — crossing it signals a sustained shift, not just a single noisy entry.
      </div>
      <canvas id="cusumChart" style="max-height:220px;"></canvas>
    </div>

    <hr class="divider">

    <div class="section">
      <div class="section-label">What's driving that signal</div>
      <div class="chart-label" style="margin-bottom:14px;">
        Four different ways of looking for unusual patterns, shown separately so each one is easy to read on its own. When they agree, that's a stronger signal.
      </div>
      <div class="chart-grid">
        <div class="chart-wrap">
          <div class="chart-label">Mahalanobis</div>
          <canvas id="detectorChartMahalanobis"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label">Copula</div>
          <canvas id="detectorChartCopula"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label">Isolation forest</div>
          <canvas id="detectorChartIsolationForest"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label">KNN</div>
          <canvas id="detectorChartKnn"></canvas>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
let charts = {};

function setStatus(msg, loading=false) {
  document.getElementById("status").innerHTML =
    (loading ? '<span class="spinner"></span>' : '') + msg;
}

function runDemo(atRisk) {
  const uid = atRisk ? "demo_atrisk" : "demo_healthy";
  document.getElementById("userId").value = uid;
  callApi(null, uid);
}

function submitFile() {
  const file = document.getElementById("fileInput").files[0];
  const uid  = document.getElementById("userId").value.trim() || "user_demo";
  callApi(file, uid);
}

function callApi(file, uid) {
  setStatus("Reading through your entries — this takes a minute or two…", true);
  document.getElementById("results").style.display = "none";
  const fd = new FormData();
  fd.append("user_id", uid);
  if (file) fd.append("file", file);
  else fd.append("demo", "true");

  fetch("/run", { method: "POST", body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) { setStatus("Something went wrong: " + d.error); return; }
      setStatus("Here's what we found.");
      renderResults(d);
    })
    .catch(e => setStatus("Something went wrong: " + e));
}

function mkLine(id, labels, datasets, yMin=null, yMax=null) {
  if (charts[id]) charts[id].destroy();
  const scales = {
    x: { ticks: { color:"#4a4a4a", font:{size:10}, maxTicksLimit:7 }, grid: { color:"#161616" } },
    y: { ticks: { color:"#4a4a4a", font:{size:10} }, grid: { color:"#161616" } }
  };
  if (yMin !== null) scales.y.min = yMin;
  if (yMax !== null) scales.y.max = yMax;
  charts[id] = new Chart(document.getElementById(id).getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color:"#555", font:{size:11} } } },
      scales
    }
  });
}

function mkBar(id, labels, datasets) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id).getContext("2d"), {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color:"#555", font:{size:11} } } },
      scales: {
        x: { ticks: { color:"#4a4a4a", font:{size:10}, maxTicksLimit:7 }, grid: { color:"#161616" } },
        y: { ticks: { color:"#4a4a4a", font:{size:10} }, grid: { color:"#161616" } }
      }
    }
  });
}

function movingAverage(arr, window=3) {
  return arr.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = arr.slice(start, i + 1);
    return slice.reduce((a,b) => a+b, 0) / slice.length;
  });
}

function renderResults(d) {
  document.getElementById("results").style.display = "block";
  const p = d.prediction;

  // Risk level
  const cls = { LOW:"col-low", MODERATE:"col-moderate", HIGH:"col-high" }[p.risk_level] || "";
  document.getElementById("riskLevel").className = "level " + cls;
  document.getElementById("riskLevel").textContent = p.risk_level;

  document.getElementById("riskStats").innerHTML = `
    <div class="stat"><div class="val col-blue">${(p.probability*100).toFixed(1)}%</div><div class="lbl">estimated risk</div></div>
    <div class="stat"><div class="val col-dim">${d.n_entries}</div><div class="lbl">entries looked at</div></div>
    <div class="stat"><div class="val ${p.intervention_recommended?'col-high':'col-low'}">${p.intervention_recommended?"Yes":"No"}</div><div class="lbl">worth a check-in</div></div>
  `;

  const note = document.getElementById("interventionNote");
  if (p.intervention_recommended) {
    note.className = "note warn";
    note.textContent = "A few signals here are worth paying attention to. It might help to talk to someone — a friend, a counsellor, or a professional you trust.";
  } else {
    note.className = "note ok";
    note.textContent = "Things look fairly steady right now. Worth keeping an eye on, as always, but nothing stands out as urgent.";
  }

  const lbl = d.timestamps;

  mkLine("sentimentChart", lbl, [{
    label:"Sentiment", data: d.sentiment_series,
    borderColor:"#4a90d9", backgroundColor:"rgba(74,144,217,0.06)",
    tension:0.4, fill:true, pointRadius:3, borderWidth:1.5
  }], -1, 1);

  mkLine("anomalyChart", lbl, [{
    label:"Anomaly Risk", data: d.anomaly_scores,
    borderColor:"#c0392b", backgroundColor:"rgba(192,57,43,0.06)",
    tension:0.4, fill:true, pointRadius:3, borderWidth:1.5
  }], 0, 1);
  
  if (d.persistent_anomaly_flags?.length) {
    const persistentDates = d.timestamps.filter((_, i) => d.persistent_anomaly_flags[i]);
  if (persistentDates.length > 0) {
    const div = document.createElement("div");
    div.style.cssText = "margin-top:8px;font-size:11px;color:#c0392b;";
    div.textContent = "⚠ Persistent signal on: " + persistentDates.join(", ");
    document.getElementById("anomalyChart").parentNode.appendChild(div);
  }
}

  if (d.calibration_status) {
    const cs = d.calibration_status;
    const banner = document.getElementById("baselineBanner");
    const title = document.getElementById("baselineTitle");
    const msg = document.getElementById("baselineMessage");

    const trendText = {
      stable: { title: "Staying steady", message: "Recent entries are consistent with this person's own typical baseline.", cls: "cusum-stable" },
      moving_away: { title: "Drifting from their own baseline", message: "Recent entries are moving further from this person's usual patterns than they were before.", cls: "cusum-upper" },
      returning_to_normal: { title: "Returning toward their baseline", message: "Recent entries are moving back closer to this person's usual patterns.", cls: "cusum-lower" },
      insufficient_data: { title: "Still calibrating", message: "Not enough entries yet to judge whether this person is drifting from their own baseline.", cls: "cusum-both" }
    }[d.baseline_trend] || { title: "—", message: "", cls: "cusum-stable" };

    banner.className = "cusum-banner " + trendText.cls;
    title.textContent = trendText.title;
    msg.textContent = trendText.message;

    document.getElementById("calibrationLabel").textContent = cs.calibrated
      ? "Baseline calibrated"
      : `Calibrating baseline: ${cs.calibration_progress} entries`;

    const pct = cs.calibrated ? 100 : Math.min(100, (cs.entries_so_far / cs.entries_needed) * 100);
    document.getElementById("calibrationFill").style.width = pct + "%";
  }

  // CUSUM status banner
  if (d.cusum_status) {
    const cs = d.cusum_status;
    const bannerCls = { stable:"cusum-stable", upper_alert:"cusum-upper",
                         lower_alert:"cusum-lower", both_alert:"cusum-both",
                         recovered:"cusum-recovered" }[cs.current_state] || "cusum-stable";
    const banner = document.getElementById("cusumBanner");
    banner.className = "cusum-banner " + bannerCls;
    document.getElementById("cusumTitle").textContent = cs.current_title;
    document.getElementById("cusumMessage").textContent = cs.current_message;
  }

  // CUSUM chart: upper/lower cumulative sums plus a constant threshold line
  if (d.cusum_upper?.length) {
    const h = d.cusum_threshold || 0;
    mkLine("cusumChart", lbl, [
      {
        label:"Upper CUSUM", data: d.cusum_upper,
        borderColor:"#c0392b", backgroundColor:"rgba(192,57,43,0.06)",
        tension:0.3, fill:false, pointRadius:2, borderWidth:1.5
      },
      {
        label:"Lower CUSUM", data: d.cusum_lower,
        borderColor:"#4a90d9", backgroundColor:"rgba(74,144,217,0.06)",
        tension:0.3, fill:false, pointRadius:2, borderWidth:1.5
      },
      {
        label:"Alert threshold (h)", data: lbl.map(()=>h),
        borderColor:"#6b6b6b", borderDash:[5,4],
        pointRadius:0, borderWidth:1, fill:false
      }
    ]);
  }

  if (d.detector_scores?.length) {
    const detectors = [
      { key: "mahalanobis",       canvas: "detectorChartMahalanobis",       color: "#4a90d9" },
      { key: "copula",            canvas: "detectorChartCopula",            color: "#c0392b" },
      { key: "isolation_forest",  canvas: "detectorChartIsolationForest",   color: "#27ae60" },
      { key: "knn",               canvas: "detectorChartKnn",               color: "#d4a017" }
    ];
    detectors.forEach(det => {
      const raw = d.detector_scores.map(s => s[det.key] || 0);
      const smoothed = movingAverage(raw, 3);
      mkLine(det.canvas, lbl, [
        {
          label: "raw", data: raw,
          borderColor: det.color, backgroundColor: "transparent",
          tension: 0.2, fill: false, pointRadius: 0, borderWidth: 1, borderDash: [2,2]
        },
        {
          label: "trend", data: smoothed,
          borderColor: det.color, backgroundColor: "transparent",
          tension: 0.35, fill: false, pointRadius: 0, borderWidth: 2.5
        }
      ], 0, 1);
    });
  }

  window.scrollTo({ top: document.getElementById("results").offsetTop - 20, behavior:"smooth" });
}

// File input
document.getElementById("fileInput").addEventListener("change", e => {
  const f = e.target.files[0];
  if (f) document.getElementById("dropLabel").innerHTML = `<strong>${f.name}</strong>`;
});

// Drag and drop
const dz = document.getElementById("dropZone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.style.borderColor="#555"; });
dz.addEventListener("dragleave", () => dz.style.borderColor="");
dz.addEventListener("drop", e => {
  e.preventDefault(); dz.style.borderColor="";
  const f = e.dataTransfer.files[0];
  if (f) {
    document.getElementById("fileInput").files = e.dataTransfer.files;
    document.getElementById("dropLabel").innerHTML = `<strong>${f.name}</strong>`;
  }
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def run():
    try:
        user_id = request.form.get("user_id", "user_demo").strip()
        demo    = request.form.get("demo", "false").lower() == "true"
        file    = request.files.get("file")

        from single_user_pipeline import run_single_user

        demo_atrisk = "atrisk" in user_id.lower() or "risk" in user_id.lower()

        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".csv",".json",".txt",".pdf",".docx",".doc"}:
                return jsonify({"error": f"Unsupported file type: {suffix}"})
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name
            tmp.close()
            file.save(tmp_path)
            result = run_single_user(user_id, file_path=tmp_path)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        else:
            result = run_single_user(user_id, use_demo=True, demo_atrisk=demo_atrisk)

        result["cusum_status"] = build_cusum_status(
            result.get("cusum_alert_upper", []),
            result.get("cusum_alert_lower", []),
            result.get("timestamps", []),
        )

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Open: http://localhost:5000")
    app.run(debug=False, port=5000)
