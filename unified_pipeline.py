import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from Stage_1.Extract_features import extract_features
from stage_2.baseline import UserBaseline
from stage_2.temporal_bin import TemporalBinning
from stage_3.tft_model import run_stage3, build_dataframe, extract_latent_and_attention
from stage_4.detectors.cusum import CUSUMDetector
from stage_4.anomaly_pipeline import MultiDetectorPipeline
from stage_4.config import PipelineConfig

try:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from scipy.special import expit
    STAGE5_AVAILABLE = True
except ImportError:
    STAGE5_AVAILABLE = False
    print(" XGBoost not installed. Stage 5 will be skipped.")

DAIC_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Stage_5")

class UnifiedJournalPipeline:
    def __init__(self, output_dir: str = "pipeline_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.user_baselines = {}
        self.temporal_binner = TemporalBinning()
        self.calibration_flags = {}
        self.raw_feature_vectors = {}

        self.tft_model = None
        self.anomaly_detector = None
        self.xgb_model = None
        self.isotonic_calibrator = None
        self.platt_params = None
        self.feature_names = None
        self.feature_vectors = {}  
        self.normalized_vectors = {}  
        self.anomaly_scores = {}  
        self.user_data = {} 
        self.user_labels = {}
        self._user_id_mapping = {}
        self.cusum_detectors = {}

        self._load_daic_model()
        
        print("Pipeline initialized")

    def _load_daic_model(self):
        """Load pretrained DAIC-WOZ XGBoost model and calibrators."""
        if not STAGE5_AVAILABLE:
            return
        model_path    = os.path.join(DAIC_MODEL_DIR, "model_new.json")
        isotonic_path = os.path.join(DAIC_MODEL_DIR, "isotonic_new.pkl")
        platt_path    = os.path.join(DAIC_MODEL_DIR, "platt_new.pkl")

        if os.path.exists(model_path):
            try:
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(model_path)
                print(f"[Stage 5] Loaded pretrained DAIC model from {model_path}")
            except Exception as e:
                print(f"[Stage 5] Failed to load DAIC model: {e}")
                self.xgb_model = None

        if os.path.exists(isotonic_path):
            try:
                with open(isotonic_path, "rb") as f:
                    self.isotonic_calibrator = pickle.load(f)
                print(f"[Stage 5] Loaded isotonic calibrator")
            except Exception as e:
                print(f"[Stage 5] Failed to load isotonic calibrator: {e}")

        if os.path.exists(platt_path):
            try:
                with open(platt_path, "rb") as f:
                    self.platt_params = pickle.load(f)
                print(f"[Stage 5] Loaded Platt calibrator (A={self.platt_params.get(chr(65)):.4f})")
            except Exception as e:
                print(f"[Stage 5] Failed to load Platt calibrator: {e}")

    def _normalize_user_id(self, user_id: str) -> str:
        if user_id not in self._user_id_mapping:
            idx = len(self._user_id_mapping)
            self._user_id_mapping[user_id] = f"user_{idx}"
        return self._user_id_mapping[user_id]

    def extract_user_entry(
        self,
        user_id: str,
        text: str,
        timestamp: Optional[datetime] = None,
        audio_path: Optional[str] = None,
        sleep_hours: Optional[float] = None,
        sleep_quality: Optional[float] = None,
        activity_level: Optional[float] = None,
        music_mood_score: Optional[float] = None,
        prev_timestamp: Optional[datetime] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            feature_vec, readable_data = extract_features(
                text=text,
                timestamp=timestamp,
                prev_timestamp=prev_timestamp,
                audio_path=audio_path,
                sleep_hours=sleep_hours,
                sleep_quality=sleep_quality,
                activity_level=activity_level,
                music_mood_score=music_mood_score
            )
            
            assert feature_vec.shape[0] == 466, f"Expected 466 features, got {feature_vec.shape[0]}"
            assert not np.any(np.isnan(feature_vec)), "NaN values detected in feature vector"
            
            if user_id not in self.feature_vectors:
                self.feature_vectors[user_id] = []
                self.user_data[user_id] = []
            
            self.feature_vectors[user_id].append(feature_vec)
            self.user_data[user_id].append({
                "timestamp": timestamp,
                "text_length": len(text),
                "has_audio": audio_path is not None
            })
            
            print(f"Stage 1 complete for user {user_id}: extracted 466 features")
            
            return feature_vec, readable_data
            
        except Exception as e:
            print(f"Stage 1 error for user {user_id}: {str(e)}")
            raise
 
    def normalize_features(
        self,
        feature_vec: np.ndarray,
        user_id: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        try:
            if user_id not in self.user_baselines:
                self.user_baselines[user_id] = UserBaseline(user_id=user_id)

            baseline = self.user_baselines[user_id]
            baseline.add_entry(feature_vec)

            context_bin = self.temporal_binner.route_vector(feature_vec, timestamp.isoformat())

            z_scored_vec = baseline.normalise(feature_vec)
            was_calibrated = z_scored_vec is not None
            if z_scored_vec is None:
                z_scored_vec = feature_vec

            if user_id not in self.normalized_vectors:
                self.normalized_vectors[user_id] = []
            if user_id not in self.calibration_flags:
                self.calibration_flags[user_id] = []
            if user_id not in self.raw_feature_vectors:
                self.raw_feature_vectors[user_id] = []

            self.normalized_vectors[user_id].append(z_scored_vec)
            self.calibration_flags[user_id].append(was_calibrated)
            self.raw_feature_vectors[user_id].append(feature_vec.copy())

            print(f"Stage 2 complete for user {user_id}: normalized features")

            return {
                "user_id": user_id,
                "timestamp": timestamp,
                "context_bin": context_bin,
                "z_scored_vector": z_scored_vec,
                "readables": {}
            }

        except Exception as e:
            print(f"Stage 2 error for user {user_id}: {str(e)}")
            raise

    def get_batch_consistent_vectors(self, user_id: str) -> list:
        if user_id not in self.raw_feature_vectors:
            return self.normalized_vectors.get(user_id, [])

        baseline = self.user_baselines.get(user_id)
        if baseline is None or not baseline.calibrated:
            return self.normalized_vectors.get(user_id, [])

        consistent_vectors = []
        for raw_vec in self.raw_feature_vectors[user_id]:
            scored = baseline.normalise(raw_vec)
            consistent_vectors.append(scored if scored is not None else raw_vec)
        return consistent_vectors

    def train_tft_model(
        self,
        num_patches: int = 10,
        hidden_size: int = 64,
        max_epochs: int = 30,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        try:
            if not self.normalized_vectors or len(self.normalized_vectors) < 1:
                raise ValueError(
                    f"Need at least 1 user for training, got {len(self.normalized_vectors)}"
                )

            patched_data = self._create_patched_data(num_patches)
            
            print(f"Created patched data for TFT: {len(patched_data)} users")
            print(f"User ID mapping: {self._user_id_mapping}")
            
            self.tft_model = run_stage3(
                patched_data=patched_data,
                feature_dim=466,
                num_patches=num_patches,
                hidden_size=hidden_size,
                max_epochs=max_epochs,
                batch_size=batch_size
            )

            import torch
            model_path = os.path.join(self.output_dir, "tft_model.pt")
            torch.save({
                "state_dict": self.tft_model["model"].state_dict(),
                "latents":    self.tft_model["latents"],
                "attention":  self.tft_model["attention"],
                "umap_coords": self.tft_model["umap_coords"],
            }, model_path)
            
            print(f"Stage 3 complete: TFT model trained and saved")
            print(f"- Latent shape: {self.tft_model['latents'].shape}")
            print(f"- Attention shape: {self.tft_model['attention'].shape}")
            
            return self.tft_model
            
        except Exception as e:
            error_msg = str(e)
            if "Unknown category" in error_msg:
                print(f"Stage 3 error: {error_msg}")
                print("WORKAROUND: If using pytorch_forecasting, consider:")
                print("  1. Setting 'add_nan=True' in the TFT data loader")
                print("  2. Retraining TFT with dynamic test categories pre-registered")
                print("  3. Using latent features from a simpler encoder (Stage 3 bypass)")
            else:
                print(f"Stage 3 error: {error_msg}")
            raise
    
    def _create_patched_data(self, num_patches: int = 10) -> Dict[str, Any]:
        patched = {}
        import torch
        
        for user_id, vectors in self.normalized_vectors.items():
            normalized_user_id = self._normalize_user_id(user_id)
            
            vectors = np.array(vectors)
            n_vectors = len(vectors)
            
            if n_vectors < num_patches:
                padding = np.zeros((num_patches - n_vectors, vectors.shape[1]))
                vectors = np.vstack([vectors, padding])

            windows = []
            for i in range(max(1, n_vectors - num_patches + 1)):
                window = vectors[i:i + num_patches]
                if len(window) < num_patches:
                    padding = np.zeros((num_patches - len(window), vectors.shape[1]))
                    window = np.vstack([window, padding])
                windows.append(window)
            
            patched[normalized_user_id] = torch.tensor(np.array(windows), dtype=torch.float32)
        
        return patched
    
    def train_anomaly_detector(self, use_latent_features: bool = False) -> None:
        try:
            model_path     = os.path.join("calibration", "models", "stage4_detectors.pkl")
            threshold_path = os.path.join("calibration", "models", "stage4_threshold_engine.pkl")

            if os.path.exists(model_path) and os.path.exists(threshold_path):
                from stage4_deployment import Stage4DeploymentPipeline
                self.anomaly_detector = Stage4DeploymentPipeline(model_path, threshold_path)
                print("Stage 4 complete: loaded pretrained DAIC-WOZ detectors")
                return

            all_vectors = []
            for user_id in self.normalized_vectors:
                vectors = self.get_batch_consistent_vectors(user_id)
                flags = self.calibration_flags.get(user_id, [True] * len(vectors))
                calibrated_vectors = [v for v, f in zip(vectors, flags) if f]
                if calibrated_vectors:
                    all_vectors.extend(calibrated_vectors)
                else:
                    all_vectors.extend(vectors)
            X_train = np.array(all_vectors)
            print(f"Stage 4: no pretrained models found — training fresh on calibrated data, shape {X_train.shape}")

            assert X_train.shape[0] > 0, "No training data available"
            assert not np.any(np.isnan(X_train)), "NaN values in training data"

            self.anomaly_detector = MultiDetectorPipeline()
            self.anomaly_detector.fit(X_train)

            detector_path = os.path.join(self.output_dir, "anomaly_detector.pkl")
            self.anomaly_detector.save(detector_path)

            print(f"Stage 4 complete: anomaly detector trained and saved")

        except Exception as e:
            print(f"Stage 4 error: {str(e)}")
            raise

    def detect_anomalies(self, feature_vec: np.ndarray, use_latent: bool = False) -> Dict[str, Any]:
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained. Call train_anomaly_detector first.")

        try:
            X = np.array([feature_vec])

            if hasattr(self.anomaly_detector, '_detect_anomalies'):
                results = self.anomaly_detector._detect_anomalies(X)
                return {
                    "overall_risk_score": float(results["overall_risk_score"]),
                    "is_anomaly":         results["is_anomaly"],
                    "detector_scores":    results["detector_scores"],
                    "timestamp":          datetime.now().isoformat()
                }
            else:
                results = self.anomaly_detector.predict(X)
                return {
                    "overall_risk_score": float(results["overall_risk_score"][0]),
                    "is_anomaly":         results["is_anomaly"][0],
                    "detector_scores":    results["metrics_summary"][0],
                    "timestamp":          datetime.now().isoformat()
                }

        except Exception as e:
            print(f"Anomaly detection error: {str(e)}")
            raise

    def fit_and_run_cusum(self, user_id) -> list:
        score = self.anomaly_scores[user_id]
        detector = CUSUMDetector()
        detector.fit(np.array([a["overall_risk_score"] for a in score]))

        cusum_result = []
        for s in score:
            result_dict = detector.update_score(s["overall_risk_score"])
            cusum_result.append(result_dict)

        self.cusum_detectors[user_id] = detector
        
        return cusum_result



    def process_entry(
        self,
        user_id: str,
        text: str,
        timestamp: Optional[datetime] = None,
        audio_path: Optional[str] = None,
        sleep_hours: Optional[float] = None,
        sleep_quality: Optional[float] = None,
        activity_level: Optional[float] = None,
        music_mood_score: Optional[float] = None,
        prev_timestamp: Optional[datetime] = None,
        label: Optional[int] = None
    ) -> Dict[str, Any]:
        
        print(f"\n{'='*60}")
        print(f"Processing entry for user: {user_id}")
        print(f"{'='*60}")
        
        feature_vec, readable = self.extract_user_entry(
            user_id=user_id,
            text=text,
            timestamp=timestamp,
            audio_path=audio_path,
            sleep_hours=sleep_hours,
            sleep_quality=sleep_quality,
            activity_level=activity_level,
            music_mood_score=music_mood_score,
            prev_timestamp=prev_timestamp
        )

        if timestamp is None:
            timestamp = datetime.now()
        
        normalized = self.normalize_features(
            feature_vec=feature_vec,
            user_id=user_id,
            timestamp=timestamp
        )

        if label is not None:
            self.user_labels[user_id] = label
        
        result = {
            "stage_1": {
                "feature_vector_shape": feature_vec.shape,
                "readable_metrics": readable
            },
            "stage_2": {
                "normalized_vector_shape": normalized["z_scored_vector"].shape,
                "context_bin": normalized["context_bin"],
                "calibrated": self.user_baselines[user_id].calibrated
            },
            "stage_2_output": {
                "z_scored_vector": normalized["z_scored_vector"]
            },
            "status": "Stages 1-2 complete"
        }
        
        if self.anomaly_detector:
            anomaly_result = self.detect_anomalies(normalized["z_scored_vector"])

            prev_scores = self.anomaly_scores.get(user_id, [])
            if len(prev_scores) >= 1:
                prev_flag = prev_scores[-1].get("is_anomaly", False)
                curr_flag = anomaly_result.get("is_anomaly", False)
                if isinstance(prev_flag, (list, np.ndarray)):
                    prev_flag = any(prev_flag)
                if isinstance(curr_flag, (list, np.ndarray)):
                    curr_flag = any(curr_flag)
                anomaly_result["is_persistent_anomaly"] = bool(prev_flag and curr_flag)
            else:
                anomaly_result["is_persistent_anomaly"] = False

            result["stage_4"] = anomaly_result
            result["status"] += " + Stage 4"

            if user_id not in self.anomaly_scores:
                self.anomaly_scores[user_id] = []
            self.anomaly_scores[user_id].append(anomaly_result)
        
        return result
    
    def assemble_stage5_features(
        self,
        window_vectors: List[np.ndarray],
        anomaly_scores: Optional[List[Dict[str, Any]]] = None,
        recent_window: int = 30,
        delta_window: int = 3
    ) -> np.ndarray:
        window_vectors = window_vectors[-recent_window:]
        window = np.array(window_vectors)

        if window.shape[0] == 0:
            raise ValueError("No vectors in window")

        features = []

        for stat_name in ["mean", "std", "max", "min"]:
            stat_func = getattr(np, f"nan{stat_name}")
            stats = stat_func(window, axis=0)
            features.extend(stats)

        if window.shape[0] >= delta_window * 2:
            early_mean = np.nanmean(window[:delta_window], axis=0)
            late_mean  = np.nanmean(window[-delta_window:], axis=0)
            deltas = early_mean - late_mean
        else:
            deltas = np.zeros(window.shape[1])
        features.extend(deltas)

        if anomaly_scores and len(anomaly_scores) > 0:
            latest_anomaly = anomaly_scores[-1]
            anomaly_features = np.array([
                float(latest_anomaly["overall_risk_score"]),
                float(latest_anomaly["detector_scores"].get("mahalanobis", 0)),
                float(latest_anomaly["detector_scores"].get("copula", 0)),
                float(latest_anomaly["detector_scores"].get("isolation_forest", 0)),
                float(latest_anomaly["detector_scores"].get("knn", 0)),
                float(any(latest_anomaly["is_anomaly"])) if isinstance(latest_anomaly["is_anomaly"], (list, np.ndarray)) else float(latest_anomaly["is_anomaly"])
            ])
        else:
            anomaly_features = np.zeros(6)

        features.extend(anomaly_features)

        feature_vector = np.nan_to_num(np.array(features))

        print(f"Stage 5 features assembled: shape {feature_vector.shape}")

        return feature_vector
    
    def train_xgboost_classifier(
        self,
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        if not STAGE5_AVAILABLE:
            raise RuntimeError("XGBoost not installed. Install with: pip install xgboost scikit-learn")


        if self.xgb_model is not None:
            print("[Stage 5] Using pretrained DAIC-WOZ model — skipping retraining.")
            return {"model": self.xgb_model, "auroc": None, "f1": None, "n_features": None}
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, f1_score
            
            X_train_list = []
            y_train_list = []
            
            for user_id, vectors in self.normalized_vectors.items():
                if user_id not in self.user_labels:
                    continue  
                
                user_anomalies = self.anomaly_scores.get(user_id, [])
                
                feature_vec = self.assemble_stage5_features(
                    vectors,
                    user_anomalies
                )
                
                X_train_list.append(feature_vec)
                y_train_list.append(self.user_labels[user_id])
            
            if len(X_train_list) < 2:
                raise ValueError(f"Need at least 2 labeled examples, got {len(X_train_list)}")
            
            X = np.array(X_train_list)
            y = np.array(y_train_list)
            
            if len(X) < 4:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.33, random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            
            pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
            
            print(f"\nTraining XGBoost on {len(X_train)} samples...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=-1
            )
            
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]
            
            try:
                auroc = roc_auc_score(y_val, y_pred_proba)
            except ValueError:
                auroc = 0.0
                
            y_pred = (y_pred_proba >= 0.5).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            print(f"Training isotonic calibration...")
            self.isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
            self.isotonic_calibrator.fit(y_pred_proba, y_val)
            
            model_path = os.path.join(self.output_dir, "xgb_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.xgb_model, f)
            
            calibrator_path = os.path.join(self.output_dir, "isotonic_calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(self.isotonic_calibrator, f)
            
            print(f"Stage 5 complete: XGBoost model trained")
            
            return {
                "model": self.xgb_model,
                "auroc": auroc,
                "f1": f1,
                "n_features": X.shape
            }
            
        except Exception as e:
            print(f"Stage 5 error: {str(e)}")
            raise
    
    def predict_classification(
        self,
        feature_vec: np.ndarray,
        calibration: str = "platt"
    ) -> Dict[str, Any]:
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost_classifier first.")
        
        try:
            X = np.array([feature_vec])
            p_raw = float(self.xgb_model.predict_proba(X)[0, 1])
            
            if self.isotonic_calibrator and calibration == "isotonic":
                p_calibrated = float(self.isotonic_calibrator.predict([[p_raw]])[0])
            elif self.platt_params and calibration == "platt":
                from scipy.special import expit as _expit
                A = self.platt_params.get("A", 1.0)
                B = self.platt_params.get("B", 0.0)
                p_calibrated = float(_expit(A * p_raw + B))
            else:
                p_calibrated = p_raw
            
            p_calibrated = np.clip(p_calibrated, 0.0, 1.0)
            
            if p_calibrated < 0.33:
                risk_level = "LOW"
                intervention = False
            elif p_calibrated < 0.67:
                risk_level = "MODERATE"
                intervention = True
            else:
                risk_level = "HIGH"
                intervention = True
            
            return {
                "probability": p_calibrated,
                "probability_raw": p_raw,
                "risk_level": risk_level,
                "intervention_recommended": intervention,
                "prediction": 1 if p_calibrated >= 0.5 else 0
            }
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            raise
    
    def predict_complete_pipeline(
        self,
        user_id: str,
        text: str,
        timestamp: Optional[datetime] = None,
        sleep_hours: Optional[float] = None,
        sleep_quality: Optional[float] = None,
        activity_level: Optional[float] = None,
        music_mood_score: Optional[float] = None
    ) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"Complete Prediction Pipeline - All 5 Stages")
        print(f"{'='*70}\n")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            result = self.process_entry(
                user_id=user_id,
                text=text,
                timestamp=timestamp,
                sleep_hours=sleep_hours,
                sleep_quality=sleep_quality,
                activity_level=activity_level,
                music_mood_score=music_mood_score
            )
            
            if self.anomaly_detector:
                anomaly_result = self.detect_anomalies(
                    result["stage_2_output"]["z_scored_vector"]
                )
                result["stage_4"] = anomaly_result
            
            if self.xgb_model and user_id in self.normalized_vectors:
                vectors = self.normalized_vectors[user_id]
                anomalies = self.anomaly_scores.get(user_id, [])
                
                feature_vec = self.assemble_stage5_features(vectors, anomalies)
                classification = self.predict_classification(feature_vec)
                result["stage_5"] = classification
                result["stage_5_features"] = feature_vec.shape
            
            return result
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise
