import numpy as np
import pandas as pd
import json
from datetime import timedelta
from daic_loader import load_daic_labels, get_daic_window
from daic_label_generator import generate_labeled_windows
from pathlib import Path

CSV_PATH = Path("train_split_Depression_AVEC2017.csv")


FEATURE_COLS = (
            [f"sbert_{i}" for i in range(384)] + 
            [f"emotion_{e}" for e in[
                'admiration','amusement','anger','annoyance','approval',
                'caring','confusion','curiosity','desire','disappointment',
                'disapproval', 'disgust','embarrassment','excitement','fear',
                'gratitude','grief','joy','love','nervousness',
                'optimism','pride','realization','relief','remorse',
                'sadness','surprise','neutral'
            ]] + 
            ['vader_neg','vader_neu','vader_pos','vader_compound',
             'vader_min_compound','vader_max_compound','vader_std_compound'] + 
            ['ttr','mtld'] + 
            ['readability_fre', 'readability_fkgl','readability_ari'] + 
            ['first_person_singular','first_person_plural'] + 
            ['avg_sentence_length','sentence_count','word_count'] + 
            ['question_ratio','exclamation_ratio','ellipsis_ratio','caps_ratio'] + 
            ['hour_sin','hour_cos','days_gap'] + 
            ['sleep_hours','sleep_quality','activity_level','music_mood',
             'mask_sleep','mask_sleep_quality','mask_activity','mask_music'] + 
            ['audio_speech_rate','audio_pause_ratio','audio_avg_pause',
             'audio_pitch_mean','audio_pitch_std','audio_rms_mean','audio_rms_std',
             'audio_emotion_angry','audio_emotion_happy',
             'audio_emotion_neutral','audio_emotion_sad'] + 
             [f'audio_mask_{i}' for i in range(11)]
        )

def extract_window(end_date, daily_df, window_size=14):
    # if feature_cols is None:
    
    feature_cols = [c for c in daily_df.columns if c not in ["uid", "date"]]
        
    # Compute start_date
    start_date = end_date - timedelta(days=window_size)

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values(by="date")

    # filter between start_date and end_date
    filtered = daily_df[daily_df["date"].between(start_date, end_date, inclusive="left")]

    #Ensures that we get exactly 466 columns even if some are missing
    # window_df = filtered[feature_cols].reindex(columns=feature_cols)
    window_df = filtered.reindex(columns=feature_cols)
    n_days = window_df.shape[0]
    # window = window_df.to_numpy()

    if n_days == 0:
        return None
    elif n_days < window_size:
        # Padding at the top if the data is lesser
        pad = np.zeros((window_size - n_days, len(feature_cols)))
        window = np.vstack([pad, window_df.to_numpy()])
        return window
    elif n_days > window_size:
        return (window_df.to_numpy()[-window_size:])
    else:
        return window_df.to_numpy()

def is_window_valid(window):
    # Robust check when missing-value strategy is unknown

    window = np.array(window)

    # Count rows that are not completely empty
    non_empty_rows = np.sum(~np.all(
        (np.isnan(window) | (window == 0)), axis=1
    ))

    if non_empty_rows >= 7:
        # also ensure there is some variance (not constant padding)
        std_per_feature = np.nanstd(window, axis=0)
        return bool(np.any(std_per_feature > 1e-6))

    return False

def assemble_feature_vector(window, anomaly_scores=None,feature_cols=FEATURE_COLS):
    # Contains the average of all the things that the patient have done

    parts = []
    names = []

    for stat in ["mean", "std", "max", "min"]:
        stats = getattr(np, f"nan{stat}")(window, axis=0)
        parts.extend(stats)
        names.extend([f"{stat}_{col}" for col in feature_cols])
    
    # trend deltas
    deltas = np.nanmean(window[:3], axis=0) - np.nanmean(window[-3:], axis=0)
    parts.extend(deltas)
    names.extend([f"delta_{col}" for col in feature_cols])

     # Part C: anomaly scores
    if anomaly_scores is None:
        anomaly_vec = np.zeros(6)
    else:
        detail = anomaly_scores["metrics_summary"][0]
        anomaly_vec = np.array([
            float(anomaly_scores["overall_risk_score"][0]),
            float(detail["mahalanobis"]),
            float(detail["copula"]),
            float(detail["isolation_forest"]),
            float(detail["knn"]),
            float(any(anomaly_scores["is_anomaly"]))
        ])
    parts.extend(anomaly_vec)

    names.extend([
        "overall_risk_score","mahalanobis","copula",
        "isolation_forest","knn","is_anomaly"
    ])
    # Overall_risk_score - combined anomaly score from multiple detectors
    # mahalanobis - anomaly score based on mahalanobis distance
    # copula - anomaly score based on copula-based detector
    # Isolation_forest - anomaly score produced by Isolation Forest detector
    # Knn - anomaly score produced by the k-nearest-neighbours detector
    # is_anomaly - binary indicator showing whether the window was flagged
    #              as anomalous 
    x_flat = np.nan_to_num(np.array(parts))
    return x_flat, names


def build_matrix():
    # Daic 
    X_rows, y_rows, sources, uids = [],[],[],[]
    feat_names = None
    daic_train = load_daic_labels(CSV_PATH)
    daic_windows_list = generate_labeled_windows(daic_train)
    TRANSCRIPT_DIR = Path("Diac-Woz")

    for item in daic_windows_list:
        uid = item["uid"]
        label = item["label"]

        for offset in [0, 5, 10, 15]:
            window = get_daic_window(uid, TRANSCRIPT_DIR, offset=offset)

            if window is None or not is_window_valid(window):
                continue

            x_flat, feat_names = assemble_feature_vector(
                window,
                anomaly_scores=None,
                feature_cols=FEATURE_COLS
            )

            X_rows.append(x_flat)
            y_rows.append(label)
            sources.append("daic")
            uids.append(uid)

    print("DAIC count:", np.sum(np.array(sources) == "daic"))

    X = np.array(X_rows)
    y = np.array(y_rows)
    sources = np.array(sources)

    print("X shape:", X.shape)
    print("Positive:", np.sum(y == 1), "Negative:", np.sum(y == 0))
    print("DAIC count:", np.sum(sources == "daic"))
    print("x_flat shape:", x_flat.shape)
    print("feature names:", len(feat_names))

    return X, y, sources, feat_names,np.array(uids)


def save_matrix(X,y,sources,feat_names,uids):
    np.save("X_train.npy",X)
    np.save("y_train.npy",y)
    np.save("sources.npy",sources)
    np.save("uids.npy", uids)
    # For developer 5 
    with open("feature_names.json", "w", encoding="utf-8") as file:
        json.dump(feat_names, file, indent=4)
    
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"pos_weight = {pos_weight}")

if __name__ == "__main__":
    X, y, sources, feat_names, uids = build_matrix()
    save_matrix(X, y, sources, feat_names,uids)
