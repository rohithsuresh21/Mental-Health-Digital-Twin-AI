import pyreadr
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\Admin\Downloads\data\dataset_rds\dataset_rds")

def load_phq():
    read_rds = pyreadr.read_r((BASE/"survey"/"PHQ-9.Rds"))
    ANSWER_MAP = {"Not at all" : 0,
                  "Several days": 1,
                  "More than half the days": 2,
                  "Nearly every day": 3}
    
    df = read_rds[None]

    column_data = [f'Q{i}' for i in range(1, 10)] # Contians 10 question and maps to numbers

    for col in column_data:
        df[col] = df[col].map(ANSWER_MAP)

    df["phq_total"] = (df[column_data]).sum(axis=1)

    #Create binary labels
    df["label"] = (df["phq_total"] >= 10).astype(int)
    
    cols_to_keep = ['uid', 'type', 'phq_total', 'label']
    df = df[cols_to_keep]

    return df

def load_activity():
    # Sitting - 0
    # Walking - 1
    # Running - 2 
    read_activity = pyreadr.read_r(BASE/"sensing"/"activity.Rds")
    df = read_activity[None]

    df["date"] = pd.to_datetime(
        df["timestamp"], # YYYY-MM-DD
          unit='s').dt.date

    df = df.groupby(["uid", "date"]).agg(
        mean_activity = (
            "activity_inference",
            "mean"
        ),
        stationary_frac=(
            "activity_inference",
            lambda x:(x == 0).mean()
        ),
        active_frac = (
            "activity_inference",
            lambda x:x.isin([1,2]).mean()
        )
    ).reset_index()
    
    return df

def load_mood():
    read_mood = pyreadr.read_r(BASE/"EMA"/"Mood.Rds")
    df = read_mood[None]

    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit='s'
    ).dt.date

    df["happy"] = pd.to_numeric(df["happy"], errors = "coerce") # Converts the missing to NaN

    df["sad"] = pd.to_numeric(df["sad"], errors = "coerce")

    df = df.groupby(["uid","date"]).agg(
        mean_happy = ("happy","mean"),
        mean_sad = ("sad", "mean")
    ).reset_index()
       
    df["net_mood"] = df["mean_happy"] - df["mean_sad"]

    return df

def load_sleep():
    read_sleep = pyreadr.read_r(BASE/"EMA"/"Sleep.Rds")
    df = read_sleep[None]

    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit='s'
        ).dt.date
    
    df["rate"] = pd.to_numeric(df["rate"],errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"],errors="coerce") # Converts to Nan

    df = df.groupby(["uid","date"]).agg(
        mean_sleep_rate = ("rate","mean"),
        mean_wake_hour = ("hour","mean")
    ).reset_index()
    
    return df
    
def merge_daily(uid, act_df, mood_df, sleep_df):

    activity = act_df[act_df["uid"] == uid].copy()
    mood = mood_df[mood_df["uid"] == uid].copy()
    sleep = sleep_df[sleep_df["uid"] == uid].copy()

    df = pd.merge( # Checks for the same uid and date
        activity,
        mood,
        on=["uid","date"],
        how="outer"
    )

    df = pd.merge(
        df,
        sleep,
        on=["uid","date"],
        how="outer"
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df 

