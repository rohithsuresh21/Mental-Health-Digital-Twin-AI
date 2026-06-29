import os, json, sys, traceback, tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

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
      <div class="section-label">What's driving that signal</div>
      <div class="chart-label" style="margin-bottom:14px;">
        Four different ways of looking for unusual patterns, shown side by side. When they agree, that's a stronger signal.
      </div>
      <canvas id="detectorChart" style="max-height:220px;"></canvas>
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

  if (d.detector_scores?.length) {
    const keys = ["mahalanobis","copula","isolation_forest","knn"];
    const cols = ["#4a90d9","#c0392b","#27ae60","#d4a017"];
    mkLine("detectorChart", lbl, keys.map((k,i)=>({
      label:k.replace("_"," "), data:d.detector_scores.map(s=>s[k]||0),
      borderColor:cols[i], tension:0.4, fill:false, pointRadius:3, borderWidth:1.5
    })));
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

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Open: http://localhost:5000")
    app.run(debug=False, port=5000)