# Mental Health Digital Twin Pipeline
 
**A multimodal anomaly detection system for mental health risk stratification**
 
IIT Indore Summer of Code 2026 (IITISoC)
 
---
 
## Overview
 
The Mental Health Digital Twin is a **5-stage AI/ML pipeline** that analyzes user-generated text, audio, activity, and sleep data to detect mental health anomalies and predict clinical risk levels. The system combines statistical and machine learning techniques to provide robust, interpretable risk assessments.
 
### Key Features
 
- **5-Stage Pipeline**: Feature extraction → Normalization → Temporal modeling → Anomaly detection → Risk classification
- **Multi-Format Input**: CSV, JSON, TXT, PDF, DOCX
- **Ensemble Anomaly Detection**: 4 independent detectors (Mahalanobis, Copula, Isolation Forest, KNN)
- **Web Interface**: Flask app with interactive dashboard
- **Interpretable Output**: Individual detector scores + risk probabilities
---
 
## Architecture
 
```
INPUT DATA (Text, Audio, Sleep, Activity, Music)
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    Stage 1: Feature Extraction (466-dim)
                     ↓
    Stage 2: Normalization (Z-score)
                     ↓
    Stage 3: Temporal Modeling (TFT)
                     ↓
    Stage 4: Anomaly Detection (Ensemble)
                     ↓
    Stage 5: Risk Classification (XGBoost)
                     ↓
    OUTPUT: LOW / MODERATE / HIGH Risk
```
 
### Stage Breakdown

| Stage | Name | Input | Output | Method |
|-------|------|-------|--------|--------|
| **1** | Feature Extraction | Raw text/audio/data | 466-dim vector | SBERT + emotion classifiers + librosa |
| **2** | Normalization | 466-dim vector | Normalized vector | Z-score with temporal context |
| **3** | Temporal Modeling | Normalized sequences | Latent representations | Temporal Fusion Transformer |
| **4** | Anomaly Detection | Normalized vectors | Anomaly scores [0,1] | Mahalanobis + Copula + IForest + KNN |
| **5** | Risk Classification | Features + anomalies | Risk label + probability | XGBoost classifier |

---

## Input Format

### Supported Formats
- **CSV** — Standard tabular (text, timestamp, sleep_hours, activity_level, etc.)
- **JSON** — Single or array of objects
- **TXT** — One entry per line
- **PDF** — Text extraction from pages
- **DOCX** — Paragraph extraction

### Required Fields
- `text` — Journal entry or user message

### Example CSV Structure
```
text,timestamp,sleep_hours,sleep_quality,activity_level,music_mood_score
"Felt good today",2024-01-15 09:30:00,8.0,0.8,0.7,0.6
"Struggled today",2024-01-16 10:15:00,5.5,0.3,0.2,0.2
```

### Optional Fields
- `timestamp` — Date/time of entry
- `sleep_hours` — Hours slept (0-24)
- `sleep_quality` — Quality rating (0-1)
- `activity_level` — Activity score (0-1)
- `music_mood_score` — Music mood score (0-1)

---




## Detector Details

### 1. Mahalanobis Distance
Statistical distance from baseline distribution using covariance matrix
- **Detects**: Distance outliers
- **Input**: Covariance matrix, mean vector

### 2. Gaussian Copula
Dependency structure modeling via marginal CDFs
- **Detects**: Feature correlation breaks
- **Input**: Correlation matrix, empirical CDFs

### 3. Isolation Forest
Tree-based anomaly scoring via recursive partitioning
- **Detects**: Isolation patterns and unusual combinations
- **Input**: Raw features, 100 trees

### 4. KNN Distance
Neighborhood density detection
- **Detects**: Local density anomalies
- **Input**: Reference dataset, k=5 neighbors

### Ensemble Aggregation
Overall Risk scoring rules operate under a soft voting system with equal weights.

Decision threshold mappings:
- Risk > 0.5 → HIGH
- 0.3 ≤ Risk ≤ 0.5 → MODERATE
- Risk < 0.3 → LOW

---

## Feature Extraction (Stage 1)

### Text Features
- SBERT embeddings (384-dim)
- VADER sentiment score
- 7-class emotion classification

### Audio Features
- Wav2Vec2 emotion classification (7 classes)
- Spectral centroid, MFCC, zero-crossing rate

### Health Features
- Sleep hours and quality
- Activity level
- Music mood score

**Total**: 466-dimensional feature vector per entry

---

## Normalization (Stage 2)

- Per-user z-score normalization
- 6-bin temporal context windows
- Handles missing values via forward-fill

---

## Temporal Modeling (Stage 3)

- **Model**: Temporal Fusion Transformer
- **Input**: Sequences of normalized vectors (10 time steps × 466 features)
- **Output**: Latent representations (64-dim) + attention weights

---

## Anomaly Detection (Stage 4)

Four independent detectors:
1. **Mahalanobis**: Covariance-based statistical distance
2. **Copula**: Marginal transformation + joint density
3. **Isolation Forest**: Path-length anomaly scoring
4. **KNN**: Average k-neighbor distance

Soft voting aggregation with equal weights

---

## Risk Classification (Stage 5)

- **Model**: XGBoost classifier
- **Training**: DAIC dataset (depressed vs. control)
- **Output**: Risk level (LOW/MODERATE/HIGH) + probability distribution
- **Calibration**: Platt & Temperature Scaling

---
## Output Format

### Anomaly Detection Result
The system calculates an overall risk score alongside individual detector breakdowns to determine whether a given data point is an outlier:
* **Overall Risk Score:** The pipeline generates a unified, soft-voting metric of **0.4368**.
* **Anomaly Status:** The system concludes that this specific entry is **not** an anomaly.
* **Detector Matrix:**
  * **Mahalanobis Distance:** Captures a statistical covariance-based distance score of **0.4357**.
  * **Gaussian Copula:** Identifies a feature correlation break score of **0.5805**.
  * **Isolation Forest:** Estimates a recursive tree-partitioning path score of **0.4366**.
  * **K-Nearest Neighbors (KNN):** Computes a neighborhood local density distance score of **0.3812**.



### Risk Classification Result
The continuous features and anomaly scores are mapped to a final clinical risk category using an calibrated classifier:
* **Assigned Risk Level:** The overall severity threshold is classified as **MODERATE**.
* **Risk Score:** The exact underlying metric is tracked at **0.4368**.
* **Probability Distribution:**
  * There is a **35% confidence** that the user's status represents a **LOW** risk level.
  * There is a **48% confidence** that the user's status represents a **MODERATE** risk level.
  * There is a **17% confidence** that the user's status represents a **HIGH** risk level.


## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_patches` | 10 | Temporal windows for TFT |
| `hidden_size` | 64 | Latent dimension |
| `max_epochs` | 30 | Training epochs |
| `batch_size` | 64 | Batch size |
| `k_neighbors` | 5 | KNN neighbors |
| `n_trees` | 100 | Isolation Forest trees |


---
## Requirements
 
`requirements.txt` for full dependency list
Key libraries:
- PyTorch 2.6.0
- Sentence-Transformers 3.4.1
- scikit-learn 1.6.1
- XGBoost 2.1.4
- librosa 0.11.0
- Flask 3.1.1



---

## Project Details

**Institution**: IIT Indore  
**Program**: Summer of Code 2026 (IITISoC)  
**Team**: AIML_25
**Status**: Under-Progress

---

## License

MIT License
