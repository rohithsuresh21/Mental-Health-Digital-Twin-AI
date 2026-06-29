import os, json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


HEALTHY = [
    "Had a great day at work, got everything done and even had time to catch up with a friend over coffee.",
    "Morning run felt amazing. Energy is high and I feel ready for whatever comes.",
    "Cooked dinner for myself and actually enjoyed it. Small wins matter.",
    "Good sleep last night. Woke up feeling refreshed and calm.",
    "Productive afternoon. Finished the report and feeling proud of myself.",
    "Had a laugh with the team today. Work doesn't feel like work when you enjoy it.",
    "Spent the evening reading. Mind feels clear and at peace.",
    "Grateful for today. Nothing special happened but I feel content.",
    "Went for a long walk. Nature helps me reset and think clearly.",
    "Called my mom today. Always lifts my mood instantly.",
    "Good progress on my goals this week. Feeling motivated to keep going.",
    "Had a nice conversation with a colleague. Feeling connected and valued.",
    "Exercised after work. Body feels tired but mind feels great.",
    "The weekend was exactly what I needed. Relaxed and recharged.",
]

AT_RISK = [
    "Can't get out of bed today. Everything feels pointless and heavy.",
    "Didn't sleep at all last night. Keep thinking about everything that's wrong.",
    "I don't see the point anymore. Just going through the motions every day.",
    "Feeling completely alone even when I'm surrounded by people.",
    "Everything is too much. I can't focus on anything for more than a minute.",
    "Cancelled plans again. Just couldn't face being around people today.",
    "I keep crying and I don't even know why. Just feel empty inside.",
    "Didn't eat much today. No appetite. Don't really care.",
    "My mind won't stop. Anxious about everything and nothing at the same time.",
    "Feel like I'm disappearing. Like no one would notice if I just stopped showing up.",
    "Skipped work again. Couldn't get myself to leave the house.",
    "Feeling hopeless. Like things will never get better no matter what I do.",
    "Dark thoughts today. Tried to distract myself but they keep coming back.",
    "I hate feeling like this. I'm so tired of feeling like this.",
]


def _sf(val):
    try:
        return float(val) if str(val).strip() not in ("","None","nan","NaN") else None
    except Exception:
        return None


def _pts(val):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except Exception:
            pass
    return datetime.now()


def load_any_file(path: str) -> list[dict]:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [{
            "text":             r.get("text", r.get("journal", r.get("entry", r.get("message","")))).strip(),
            "timestamp":        _pts(r.get("timestamp", r.get("date", ""))),
            "sleep_hours":      _sf(r.get("sleep_hours")),
            "sleep_quality":    _sf(r.get("sleep_quality")),
            "activity_level":   _sf(r.get("activity_level")),
            "music_mood_score": _sf(r.get("music_mood_score")),
        } for r in rows if r.get("text","").strip()]

    elif suffix == ".json":
        import json as _json
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [{
            "text":             d.get("text", d.get("journal", d.get("entry",""))).strip(),
            "timestamp":        _pts(d.get("timestamp", d.get("date",""))),
            "sleep_hours":      _sf(d.get("sleep_hours")),
            "sleep_quality":    _sf(d.get("sleep_quality")),
            "activity_level":   _sf(d.get("activity_level")),
            "music_mood_score": _sf(d.get("music_mood_score")),
        } for d in data if d.get("text","").strip()]

    elif suffix == ".txt":
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8",errors="ignore").splitlines() if l.strip()]
        base  = datetime.now() - timedelta(days=len(lines))
        return [{"text": l, "timestamp": base + timedelta(days=i),
                 "sleep_hours": None, "sleep_quality": None,
                 "activity_level": None, "music_mood_score": None}
                for i, l in enumerate(lines)]

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        base  = datetime.now() - timedelta(days=len(lines))
        return [{"text": l, "timestamp": base + timedelta(days=i),
                 "sleep_hours": None, "sleep_quality": None,
                 "activity_level": None, "music_mood_score": None}
                for i, l in enumerate(lines)]

    elif suffix in (".docx", ".doc"):
        try:
            import docx
        except ImportError:
            raise ImportError("pip install python-docx")
        doc   = docx.Document(path)
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        base  = datetime.now() - timedelta(days=len(lines))
        return [{"text": l, "timestamp": base + timedelta(days=i),
                 "sleep_hours": None, "sleep_quality": None,
                 "activity_level": None, "music_mood_score": None}
                for i, l in enumerate(lines)]

    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _make_demo_records(is_atrisk: bool, n_days: int = 14) -> list[dict]:
    random.seed(42)
    pool = AT_RISK if is_atrisk else HEALTHY
    base = datetime.now() - timedelta(days=n_days)
    records = []
    for i in range(n_days):
        ts = base + timedelta(days=i, hours=random.randint(0, 23),
                              seconds=random.randint(0, 1800))
        records.append({
            "text":             pool[i % len(pool)],
            "timestamp":        ts,
            "sleep_hours":      round(random.uniform(2.0, 5.5 if is_atrisk else 9.0), 1),
            "sleep_quality":    round(random.uniform(0.0, 0.35 if is_atrisk else 1.0), 2),
            "activity_level":   round(random.uniform(0.0, 0.25 if is_atrisk else 1.0), 2),
            "music_mood_score": round(random.uniform(0.0, 0.3  if is_atrisk else 1.0), 2),
        })
    return records


def run_single_user(user_id: str, file_path: Optional[str] = None,
                    use_demo: bool = False, demo_atrisk: bool = False) -> dict:
    from unified_pipeline import UnifiedJournalPipeline

    pipeline = UnifiedJournalPipeline()

    if file_path and Path(file_path).exists():
        records = load_any_file(file_path)
        records = [r for r in records if r["text"]]
        print(f"Loaded {len(records)} entries from {file_path}")
    else:
        records = _make_demo_records(demo_atrisk)
        print(f"Using 14-day demo ({'at-risk' if demo_atrisk else 'healthy'}) data for {user_id}")

    if len(records) < 3:
        raise ValueError(f"Need at least 3 entries, got {len(records)}")

    prev_ts = None
    sentiment_series, sleep_series, activity_series, music_series = [], [], [], []
    emotions_series, timestamps = [], []

    for rec in records:
        result = pipeline.process_entry(
            user_id=user_id,
            text=rec["text"],
            timestamp=rec["timestamp"],
            prev_timestamp=prev_ts,
            sleep_hours=rec["sleep_hours"],
            sleep_quality=rec["sleep_quality"],
            activity_level=rec["activity_level"],
            music_mood_score=rec["music_mood_score"],
        )
        prev_ts = rec["timestamp"]
        m = result["stage_1"]["readable_metrics"]["raw_display_metrics"]
        sentiment_series.append(round(m["sentiment_score"], 4))
        emotions_series.append(m["dominant_emotion"])
        timestamps.append(rec["timestamp"])
        if rec["sleep_hours"] is not None:
            sleep_series.append((rec["timestamp"], rec["sleep_hours"]))
        if rec["activity_level"] is not None:
            activity_series.append((rec["timestamp"], rec["activity_level"]))
        if rec["music_mood_score"] is not None:
            music_series.append((rec["timestamp"], rec["music_mood_score"]))

    # Train TFT and anomaly detector on user data only (no fake reference users)
    n = len(records)
    tft = pipeline.train_tft_model(
        num_patches=min(10, max(3, n - 1)),
        hidden_size=32,
        max_epochs=5,
        batch_size=8,
    )

    pipeline.train_anomaly_detector(use_latent_features=False)
    anomaly_results = []
    for vec in pipeline.normalized_vectors[user_id]:
        anomaly_results.append(pipeline.detect_anomalies(vec))

    # Store anomaly results
    pipeline.anomaly_scores[user_id] = anomaly_results

    # XGBoost uses pretrained DAIC model — no retraining needed
    xgb = pipeline.train_xgboost_classifier()

    vecs      = pipeline.normalized_vectors[user_id]
    anomalies = pipeline.anomaly_scores.get(user_id, [])
    features  = pipeline.assemble_stage5_features(vecs, anomalies)
    prediction = pipeline.predict_classification(features)

    return {
        "user_id":          user_id,
        "n_entries":        len(records),
        "timestamps":       [t.strftime("%Y-%m-%d") for t in timestamps],
        "sentiment_series": sentiment_series,
        "emotions_series":  emotions_series,
        "sleep_series":     [(t.strftime("%Y-%m-%d"), v) for t, v in sleep_series],
        "activity_series":  [(t.strftime("%Y-%m-%d"), v) for t, v in activity_series],
        "music_series":     [(t.strftime("%Y-%m-%d"), v) for t, v in music_series],
        "anomaly_scores":   [round(a["overall_risk_score"], 4) for a in anomaly_results],
        "detector_scores":  [a["detector_scores"] for a in anomaly_results],
        "prediction":       prediction,
        "tft_latent_shape": list(tft["latents"].shape),
        "xgb_auroc":        round(xgb["auroc"], 4) if xgb["auroc"] is not None and xgb["auroc"] == xgb["auroc"] else 0.0,
    }


if __name__ == "__main__":
    r = run_single_user("demo_healthy", use_demo=True, demo_atrisk=False)
    print(json.dumps({k: v for k, v in r.items() if k != "detector_scores"}, indent=2, default=str))