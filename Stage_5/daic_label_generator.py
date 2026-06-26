import pandas as pd
import numpy as np
from datetime import timedelta


def generate_labeled_windows(df):
    end_date = pd.Timestamp("2017-01-01").date()

    windows = []

    for _, row in df.iterrows():
        windows.append({
            "uid": row["uid"],
            "end_date": end_date,
            "label": row["label"]
        })

    return windows
    