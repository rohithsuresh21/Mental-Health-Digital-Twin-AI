import pandas as pd
import numpy as np
from pathlib import Path
from Extract_features import extract_features

def load_daic_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["Participant_ID","PHQ8_Binary"]]
    df.columns = ["uid", "label"] # Renamed
    return df

def load_transcript(participant_id, transcript_dir):
    matches = list(
        Path(transcript_dir).rglob(f"{participant_id}_TRANSCRIPT.csv")
    )

    if len(matches) == 0:
        return []

    path = matches[0]
    
    read_csv = pd.read_csv(path, sep='\t') # tab seperated
    # Filtering the only the words of the patient
    filtered = read_csv[read_csv["speaker"] == "Participant"]
    # Drop all the NAN and return as list with string
    result = filtered['value'].dropna().astype(str).tolist()

    return result


def get_daic_window(participant_id, transcript_dir):
    transcript = load_transcript(participant_id, transcript_dir)

    if len(transcript) == 0:
        return None
    # Convertig the whole into 14 day series data
    
    n_bins = np.round(np.linspace(0, len(transcript), num=15)).astype(int)

    temp = []

    for i in range(14):
        start = n_bins[i]
        end = n_bins[i+1]
        temp.append(" ".join(transcript[start:end]))
    # Adding Dev1's encoder
    features = []
    for row in temp:
        feature_vec, _ = extract_features(row)
        features.append(feature_vec)

    return np.array(features)


