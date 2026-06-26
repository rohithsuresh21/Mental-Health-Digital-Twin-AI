import pandas as pd
import numpy as np
from datetime import timedelta

def generate_labeled_windows(phq_df):
    term_start = pd.Timestamp("2013-03-27").date()

    window = []

    for _, row in phq_df.iterrows():
        uid = row["uid"] # Returns the value inside the column 
        label = row["label"]
        if row["type"] == "pre":
            end_date = term_start + timedelta(days=14) # First 14 days
        else:
            end_date = term_start + timedelta(days=69) # Last 14 days

        window.append({
            "uid":uid,
            "end_date":end_date,
            "label":label
        })
    
    return window

def generate_ordinal_label(window_array):
    # In case of the people with the phq - 9 labels missing
    # Compute the baseline of the people(first 5 days)
    baseline = np.nanmean(window_array[:5], axis=0)
    
    # Compute the end_state of the people
    end_state = np.nanmean(window_array[-5:], axis=0)

    # Compute the deviation of the persons behaviour
    deviation = end_state - baseline

    #Compute the euclidien distance
    dist = np.linalg.norm(deviation) / (np.nanstd(window_array) + 1e-8)

    if dist < 1.0:
        return 0
    elif dist < 2.0:
        return 1
    elif dist < 3.0:
        return 2
    else:
        return 3
    
