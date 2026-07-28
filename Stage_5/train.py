import json
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.metrics import f1_score

# Loading the dataset
X = np.load("X_train.npy")
y = np.load("y_train.npy")
uids = np.load("uids.npy")

# Split into train and val(80/20)
unique_uids = np.unique(uids)

# One label per participant
participant_labels = []
for uid in unique_uids:
    participant_labels.append(y[uids == uid][0])

participant_labels = np.array(participant_labels)

train_uids, val_uids = train_test_split(
    unique_uids,
    test_size=0.2,
    random_state=42,
    stratify=participant_labels
)

train_mask = np.isin(uids, train_uids)
val_mask = np.isin(uids, val_uids)

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

print(f"Train participants: {len(train_uids)}")
print(f"Validation participants: {len(val_uids)}")
print(f"Train windows: {len(X_train)}")
print(f"Validation windows: {len(X_val)}")
# Train XGBoost
pos_weight = (y_train==0).sum() / max(1, (y_train==1).sum())

model = xgb.XGBClassifier(
    n_estimators = 150, # No. of tree
    max_depth = 3, 
    learning_rate = 0.03,
    subsample = 0.7, # Fraction of train data each tree sees
    colsample_bytree = 0.7, # Fraction of feature used
    gamma = 0.5,
    reg_alpha = 1.0,
    reg_lambda = 3.0,
    min_child_weight = 5,
    scale_pos_weight = pos_weight,
    objective = "binary:logistic",
    eval_metric = "auc",
    early_stopping_rounds = 30,
    random_state = 42, # reproducability
    n_jobs = -1 # Use all cores of cpu
)

# Fit model
model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    verbose = 50,
)


val_probs = model.predict_proba(X_val)[:, 1]

val_uids_windows = uids[val_mask]

participant_probs = []
participant_labels = []

for uid in np.unique(val_uids_windows):
    mask = val_uids_windows == uid

    # Average the probabilities of all windows
    participant_probs.append(val_probs[mask].mean())

    # Same label for all windows of a participant
    participant_labels.append(y_val[mask][0])

participant_probs = np.array(participant_probs)
participant_labels = np.array(participant_labels)

auroc = roc_auc_score(participant_labels, participant_probs)
print(f"Participant Validation AUROC: {auroc:.4f}")

participant_preds = (participant_probs >= 0.5).astype(int)
f1 = f1_score(
    participant_labels,
    participant_preds,
    zero_division=0
)

print(f"Participant Validation F1 score: {f1:.4f}")

# Keep this variable name for the remaining calibration code
val_probs = participant_probs
y_val = participant_labels

# ECE caliberation
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        # mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean() # reality
        conf = y_prob[mask].mean() # confidence
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece

ece_raw = compute_ece(y_val, val_probs)
print(f"ECE before calibration: {ece_raw:.4f}")

# Temperature scaling
def temperature_scale(probs, T):
    # Convert prob to logit, divide by T, convert back
    logits = np.log(probs / (1 - probs + 1e-7) + 1e-7)
    scaled_logits = logits / T
    return 1 / (1 + np.exp(-scaled_logits))

# Grid search T in [0.5, 2.0]
best_T = 1.0
best_ece = float('inf')
for T in np.arange(0.5, 2.0, 0.1):
    cal = temperature_scale(val_probs, T)
    ece = compute_ece(y_val, cal)
    if ece < best_ece:
        best_ece = ece
        best_T = T

cal_temp = temperature_scale(val_probs, best_T)
ece_temp = compute_ece(y_val, cal_temp)
print(f"Best T: {best_T:.2f}")
print(f"ECE after temperature scaling: {ece_temp:.4f}")

# Platt scaling calibration
eps = 1e-7
logits = np.log((val_probs + eps) / (1 - val_probs + eps)).reshape(-1,1) # to prevent out-off bound error
lr = LogisticRegression(C=1e10)
lr.fit(logits, y_val)
A = lr.coef_[0][0]
B = lr.intercept_[0]
cal_platt = expit(A * logits + B).flatten()
ece_platt = compute_ece(y_val, cal_platt)
auroc_platt = roc_auc_score(y_val, cal_platt)
print(f"AUROC after Platt: {auroc_platt:.4f}")
print(f"ECE after Platt: {ece_platt:.4f}")
print(f"A={A:.4f}, B={B:.4f}")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print(f"  Train samples:        {len(X_train)}")
print(f"  Val samples:          {len(X_val)}")
print(f"  Validation AUROC:     {auroc:.4f}")
print(f"  F1 score:             {f1:.4f}")
print(f"  ECE raw:              {ece_raw:.4f}")
print(f"  ECE temperature:         {ece_temp:.4f}")
print(f"  ECE Platt:            {ece_platt:.4f}")
print(f"  pos_weight used:      {pos_weight:.2f}")
print("="*50)

# Save
model.save_model("model.json")

with open("temperature.json", "w") as f:
    json.dump({"T": best_T}, f)
print(f"temperature.json saved — T={best_T:.2f}")

with open("platt.pkl", "wb") as f:
    pickle.dump({"A": A,"B":B}, f)

print("model.json saved")
print("temperature.json saved")
print("platt.pkl saved")



