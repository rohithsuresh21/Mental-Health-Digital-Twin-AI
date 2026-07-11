import json
import pickle
import numpy as np
import xgboost as xgb
from scipy.special import expit

_model = None
_T = None
_platt_A = None
_platt_B = None
_names = None
_pca = None
_scaler = None

def _load():
    global _model, _T,_platt_A , _platt_B, _names, _pca, _scaler
    _model = xgb.XGBClassifier()
    _model.load_model("model.json")

    with open("temperature.json") as f:
        _T = json.load(f)["T"]

    with open("platt.pkl", "rb") as f:
        d = pickle.load(f)
        _platt_A = d["A"]
        _platt_B = d["B"]

    with open("pca.pkl", "rb") as f:
        d = pickle.load(f)
        _pca = d["pca"]
        _scaler = d["scaler"]

    with open("feature_names.json") as f:
        _names = json.load(f)

def predict(X, calibration="temperature"):
    global _platt_A, _platt_B, _T, _names, _model, _scaler, _pca # or platt
    if _model is None:
        _load()

    if X.ndim == 1:
        X = np.reshape(X,(1, -1)) # -1 figures out the dimension of the input auto
    
    X = _scaler.transform(X)
    X = _pca.transform(X)

    p_raw = float(_model.predict_proba(X)[0, 1])

    if calibration == "temperature":
        # Temperature scaling
        logit = np.log(p_raw / (1 - p_raw + 1e-7) + 1e-7)
        p_cal = float(1 / (1 + np.exp(-logit / _T)))
    else:
        logit = np.log((p_raw + 1e-7) / (1 - p_raw + 1e-7))
        p_cal = float(expit(_platt_A * logit + _platt_B))
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
    global _model
    if _model is None:
        _load()
    return _model

def get_feature_names():
    global _model
    if _model is None:
        _load()
    return _names
