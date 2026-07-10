import os, json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import Counter
import random

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
        return float(val) if str(val).strip() not in ("", "None", "nan", "NaN") else None
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
            "text":             r.get("text", r.get("journal", r.get("entry", r.get("message", "")))).strip(),
            "timestamp":        _pts(r.get("timestamp", r.get("date", ""))),
            "sleep_hours":      _sf(r.get("sleep_hours")),
            "sleep_quality":    _sf(r.get("sleep_quality")),
            "activity_level":   _sf(r.get("activity_level")),
            "music_mood_score": _sf(r.get("music_mood_score")),
        } for r in rows if r.get("text", "").strip()]

    elif suffix == ".json":
        import json as _json
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [{
            "text":             d.get("text", d.get("journal", d.get("entry", ""))).strip(),
            "timestamp":        _pts(d.get("timestamp", d.get("date", ""))),
            "sleep_hours":      _sf(d.get("sleep_hours")),
            "sleep_quality":    _sf(d.get("sleep_quality")),
            "activity_level":   _sf(d.get("activity_level")),
            "music_mood_score": _sf(d.get("music_mood_score")),
        } for d in data if d.get("text", "").strip()]

    elif suffix == ".txt":
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
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
    context_bin_series = []

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
        context_bin_series.append(result["stage_2"]["context_bin"])
        if rec["sleep_hours"] is not None:
            sleep_series.append((rec["timestamp"], rec["sleep_hours"]))
        if rec["activity_level"] is not None:
            activity_series.append((rec["timestamp"], rec["activity_level"]))
        if rec["music_mood_score"] is not None:
            music_series.append((rec["timestamp"], rec["music_mood_score"]))

    n = len(records)

    if n >= 60:
        num_patches = 14
        hidden_size = 64
        max_epochs  = 10
        batch_size  = 16
    elif n >= 30:
        num_patches = 10
        hidden_size = 48
        max_epochs  = 7
        batch_size  = 12
    else:
        num_patches = min(10, max(3, n - 1))
        hidden_size = 32
        max_epochs  = 5
        batch_size  = 8

    tft = pipeline.train_tft_model(
        num_patches=num_patches,
        hidden_size=hidden_size,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    model_dir = Path("calibration/models")
    detectors_file = model_dir / "stage4_detectors.pkl"
    threshold_file = model_dir / "stage4_threshold_engine.pkl"

    all_vecs = pipeline.get_batch_consistent_vectors(user_id)
    n_total = len(all_vecs)

    if detectors_file.exists() and threshold_file.exists():
        print(f"Stage 4: Loading pretrained calibration models from {model_dir}...")
        import pickle
        with open(detectors_file, "rb") as f:
            pipeline.detectors = pickle.load(f)
        with open(threshold_file, "rb") as f:
            pipeline.threshold_engine = pickle.load(f)
    else:
        print("Stage 4: Pretrained models missing – training fresh with chronological split...")
        n_train = max(10, int(n_total * 0.7))
        train_vecs = all_vecs[:n_train]
        X_train = np.array(train_vecs)
        print(f"  Training on first {n_train}/{n_total} entries, scoring all {n_total}")

        from stage_4.anomaly_pipeline import MultiDetectorPipeline
        pipeline.anomaly_detector = MultiDetectorPipeline()
        pipeline.anomaly_detector.fit(X_train)

        import pickle as _pkl
        detector_path = os.path.join(pipeline.output_dir, "anomaly_detector.pkl")
        pipeline.anomaly_detector.save(detector_path)

    anomaly_results = []
    for vec in all_vecs:
        anomaly_results.append(pipeline.detect_anomalies(vec))

    pipeline.anomaly_scores[user_id] = anomaly_results
     
    cusum_results = pipeline.fit_and_run_cusum(user_id)
    cusum_threshold = round(float(pipeline.cusum_detectors[user_id].h), 4)

    xgb = pipeline.train_xgboost_classifier()

    vecs      = pipeline.normalized_vectors[user_id]
    anomalies = pipeline.anomaly_scores.get(user_id, [])
    if len(vecs) < 5:
        print(f"[Stage 5] Only {len(vecs)} entries — skipping classification, need at least 5.")
        prediction = {
        "probability": 0.0,
        "probability_raw": 0.0,
        "risk_level": "LOW",
        "intervention_recommended": False,
        "prediction": 0,
        "note": "insufficient entries for reliable classification"
    }
    else:
        features = pipeline.assemble_stage5_features(vecs, anomalies)
        print(f"DEBUG: n_entries={len(records)}, vecs={len(vecs)}, features_shape={features.shape}")
        prediction = pipeline.predict_classification(features)
        print(f"DEBUG: probability_raw={prediction.get('probability_raw')}, probability_calibrated={prediction.get('probability')}")

    ub = pipeline.user_baselines[user_id]
    calibration_status = ub.calibration_status()
    cutoff = ub.min_entries_to_fit - 1

    deviation_series = []
    for i, vec in enumerate(vecs):
        if i < cutoff:
            deviation_series.append(None)
        else:
            text_part = vec[ub.TEXT_START:ub.TEXT_END]
            audio_part = vec[ub.AUDIO_START:ub.AUDIO_END]
            deviation = float(np.mean(np.abs(np.concatenate([text_part, audio_part]))))
            deviation_series.append(round(deviation, 4))

    valid_deviation = [v for v in deviation_series if v is not None]
    if len(valid_deviation) >= 6:
        k = max(3, len(valid_deviation) // 3)
        early_avg = float(np.mean(valid_deviation[:k]))
        late_avg = float(np.mean(valid_deviation[-k:]))
        diff = late_avg - early_avg
        if diff > 0.15:
            baseline_trend = "moving_away"
        elif diff < -0.15:
            baseline_trend = "returning_to_normal"
        else:
            baseline_trend = "stable"
    else:
        baseline_trend = "insufficient_data"

    bin_labels = {
        "Morning_Weekday":   "Morning (Weekday)",
        "Afternoon_Weekday": "Afternoon (Weekday)",
        "Evening_Weekday":   "Evening (Weekday)",
        "Morning_Weekend":   "Morning (Weekend)",
        "Afternoon_Weekend": "Afternoon (Weekend)",
        "Evening_Weekend":   "Evening (Weekend)",
    }
    bin_counts_raw = Counter(context_bin_series)
    context_bin_counts = {bin_labels.get(k, k): bin_counts_raw.get(k, 0) for k in bin_labels}

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
        "cusum_upper":        [round(float(c["cusum_upper"]), 4) for c in cusum_results],
        "cusum_lower":        [round(float(c["cusum_lower"]), 4) for c in cusum_results],
        "cusum_alert_upper":  [bool(c["cusum_alert_upper"]) for c in cusum_results],
        "cusum_alert_lower":  [bool(c["cusum_alert_lower"]) for c in cusum_results],
        "cusum_threshold":    cusum_threshold,
        "persistent_anomaly_flags": [bool(a.get("is_persistent_anomaly", False)) for a in anomaly_results],
        "prediction":       prediction,
        "tft_latent_shape": list(tft["latents"].shape),
        "xgb_auroc":        round(xgb["auroc"], 4) if xgb["auroc"] is not None and xgb["auroc"] == xgb["auroc"] else 0.0,
        "calibration_status": calibration_status,
        "baseline_deviation_series": deviation_series,
        "baseline_trend": baseline_trend,
        "context_bin_counts": context_bin_counts,
    }


if __name__ == "__main__":
    r = run_single_user("demo_healthy", use_demo=True, demo_atrisk=False)
    print(json.dumps({k: v for k, v in r.items() if k != "detector_scores"}, indent=2, default=str))
