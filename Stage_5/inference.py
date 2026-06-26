import json
import pickle
import numpy as np
import xgboost as xgb
from scipy.special import expit

_model = None
_iso = None
_platt_A = None
_platt_B = None
_names = None

def _load():
    global _model, _iso,_platt_A , _platt_B, _names
    _model = xgb.XGBClassifier()
    _model.load_model("model.json")

    with open("isotonic.pkl", "rb") as f:
        _iso = pickle.load(f)

    with open("platt.pkl", "rb") as f:
        d = pickle.load(f)
        _platt_A = d["A"]
        _platt_B = d["B"]

    with open("feature_names.json") as f:
        _names = json.load(f)

def predict(X, calibration="isotonic"): # or platt
    if _model is None:
        _load()

    if X.ndim == 1:
        X = np.reshape(X,(1, -1)) # -1 figures out the dimension of the input auto
    
    p_raw = float(_model.predict_proba(X)[0, 1])

    if calibration == "isotonic":
        p_cal = float(_iso.predict([p_raw])[0])
    else:
        p_cal = float(expit(_platt_A * p_raw + _platt_B))
    p_cal = np.clip(p_cal, 0.0, 1.0)
    p_cal = float(p_cal)

    badge = ""
    intervene = False
    action_string = ""

    if p_cal < 0.33:
        badge = "LOW"
        intervene = False
        action_string = "Continue monitoring. No immediate action required."
    elif p_cal < 0.66:
        badge = "MODERATE"
        intervene = False
        action_string = "Review behavioral trends. Consider check-in within 7 days."
    else:
        badge = "ELEVATED"
        intervene = True
        action_string = "Recommend clinical follow-up within 48 hours."

    return {
    "p_raw": p_raw,
    "p_calibrated": p_cal,
    "risk_badge": badge,
    "intervention": intervene,
    "action": action_string,
    "label_source": "DAIC-Woz PHQ-8"
    }

def get_model_for_shap():
    _load()
    return _model

def get_feature_names():
    _load()
    return _names