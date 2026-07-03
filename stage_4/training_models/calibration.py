import numpy as np
import pickle
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage_4.detectors.mahalanobis import ProductionMahalanobisDetector
from stage_4.detectors.copula import GaussianCopulaAnomalyDetector
from stage_4.detectors.isolation_forest import ProductionIsolationForestDetector
from stage_4.detectors.knn_detector import NativeKnnDistanceDetector
from stage_4.utils.thresholds import StaticPercentileEngine
from stage_4.config import PipelineConfig

class Stage4DetectorCalibration:

    def __init__(self, data_dir: str = "calibration/stage4_training_data"):

        self.data_dir = Path(data_dir)
 
        self.healthy_vectors = None
        self.atrisk_vectors = None
        self.all_vectors = None

        self.detectors = {
            "mahalanobis": None,
            "copula": None,
            "isolation_forest": None,
            "knn": None,
        }

        self.threshold_engine = StaticPercentileEngine(
            target_percentile=PipelineConfig.THRESHOLD_PERCENTILE
            )
        
        self.statistics = {}
        
    def load_vectors(self):
        print(f"Loading vectors from {self.data_dir}...")

        healthy_file  = self.data_dir / "healthy_vectors.npy"
        atrisk_file   = self.data_dir / "atrisk_vectors.npy"    

        if not healthy_file.exists() or not atrisk_file.exists():
            raise FileNotFoundError(f"Vector files not found in {self.data_dir}. Please ensure 'healthy_vectors.npy' and 'atrisk_vectors.npy' are present.")
        
        self.healthy_vectors = np.load(healthy_file).astype(np.float64)
        self.atrisk_vectors = np.load(atrisk_file).astype(np.float64)
        self.all_vectors = np.vstack([self.healthy_vectors, self.atrisk_vectors])
 
        print(f"  Healthy : {self.healthy_vectors.shape}")
        print(f"  At-risk : {self.atrisk_vectors.shape}")
        print(f"  Combined: {self.all_vectors.shape}")

    def calibrate_mahalanobis(self):
        print("\n[1/4] Calibrating Mahalanobis detector...")
        detector = ProductionMahalanobisDetector(
            regularization=PipelineConfig.MAHALANOBIS_REGULARIZATION
        )
        detector.fit(self.healthy_vectors)
        self.detectors["mahalanobis"] = detector
        self.statistics["mahalanobis"] = self._compute_stats(detector, "Mahalanobis")


    def calibrate_copula(self):
        print("\n[2/4] Calibrating Gaussian Copula detector...")
        detector = GaussianCopulaAnomalyDetector(
            epsilon = PipelineConfig.COPULA_EPSILON
        )
        detector.fit(self.healthy_vectors)
        self.detectors["copula"] = detector
        self.statistics["copula"] = self._compute_stats(detector, "Copula")

    def calibrate_isolation_forest(self):
        print("\n[3/4] Calibrating Isolation Forest detector...")
        detector = ProductionIsolationForestDetector(
            n_estimators=PipelineConfig.ISOLATION_FOREST_N_ESTIMATORS,
            contamination=PipelineConfig.ISOLATION_FOREST_CONTAMINATION,
            random_state=PipelineConfig.RANDOM_SEED
        )
        detector.fit(self.all_vectors)
        self.detectors["isolation_forest"] = detector
        self.statistics["isolation_forest"] = self._compute_stats(detector, "IForest")

    def calibrate_knn(self):
        print("\n[4/4] Calibrating KNN detector...")
        detector = NativeKnnDistanceDetector(
            n_neighbors=PipelineConfig.KNN_N_NEIGHBORS,
            algorithm=PipelineConfig.KNN_ALGORITHM,
            metric = PipelineConfig.KNN_METRIC
        )
        detector.fit(self.healthy_vectors)
        self.detectors["knn"] = detector
        self.statistics["knn"] = self._compute_stats(detector, "KNN")

    def _compute_state(self, detector, name: str) -> dict:
        scores_healthy = detector.predict_score(self.healthy_vectors)
        scores_atrisk = detector.predict_score(self.atrisk_vectors)
 
        stats = {
            "healthy_mean": float(np.mean(scores_healthy)),
            "healthy_std": float(np.std(scores_healthy)),
            "atrisk_mean": float(np.mean(scores_atrisk)),
            "atrisk_std": float(np.std(scores_atrisk)),
            "separation": float(np.mean(scores_atrisk) - np.mean(scores_healthy)),
        }
 
        print(f"  {name}:")
        print(f"    Healthy: μ={stats['healthy_mean']:.4f}, σ={stats['healthy_std']:.4f}")
        print(f"    At-risk: μ={stats['atrisk_mean']:.4f}, σ={stats['atrisk_std']:.4f}")
        print(f"    Separation: {stats['separation']:.4f}")
 
        return stats
    

def fit_threshold_engine(self):
    print("\nFitting 95th-percentile threshold engine on healthy scores...")
    w = PipelineConfig.DETECTOR_WEIGHTS
    score_dict = {}
    for key, detector in self.detectors.items():
        if detector is None:
            raise RuntimeError(f"Detector '{key}' has not been calibrated yet.")
        score_dict[key] = detector.predict_score(self.healthy_vectors)

    self.threshold_engine.fit(score_dict)

    combined = np.array([
            score_dict["mahalanobis"] * w["mahalanobis"] +
            score_dict["copula"] * w["copula"] +
            score_dict["isolation_forest"] * w["isolation_forest"] +
            score_dict["knn"] * w["knn"]
        ])
    
    self.threshold_engine.thresholds["combined"] = float(
            np.percentile(combined, PipelineConfig.THRESHOLD_PERCENTILE)
        )
    print(f"  Thresholds set: {self.threshold_engine.thresholds}")

    def save_calibration(self, output_dir: str = "calibration/models"):
        self.fit_threshold_engine()
 
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
 
        print(f"\nSaving calibrated detectors to {output_dir}...")
 
        with open(output_dir / "stage4_detectors.pkl", "wb") as f:
            pickle.dump(self.detectors, f)
 
        with open(output_dir / "stage4_threshold_engine.pkl", "wb") as f:
            pickle.dump(self.threshold_engine, f)
 
        with open(output_dir / "detector_statistics.json", "w") as f:
            json.dump(self.statistics, f, indent=2)
 
        print("  stage4_detectors.pkl")
        print("  stage4_threshold_engine.pkl")
        print("  detector_statistics.json")
 
        return output_dir