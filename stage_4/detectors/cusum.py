import numpy as np
from stage_4.config import PipelineConfig

class CUSUMDetector:

    def __init__(self):

        self.k_multiplier = PipelineConfig.CUSUM_K_MULTIPLIER
        self.h_multiplier = PipelineConfig.CUSUM_H_MULTIPLIER

        self.cusum_upper: float = 0.0
        self.cusum_lower: float = 0.0
        self.mu_0: float = 0.0
        self.sigma: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "CUSUMDetector":

        X_arr = np.asarray(X, dtype=np.float64)
        self.mu_0 = float(np.mean(X_arr))
        self.sigma = float(np.std(X_arr))
        if self.sigma < 1e-7:
            self.sigma = 1e-7
        self.k = self.k_multiplier * self.sigma
        self.h = self.h_multiplier * self.sigma
        self.is_fitted = True
        return self
    
    def update_score(self, overall_risk_score: float) -> dict:

        if not self.is_fitted:
            raise ValueError("CUSUMDetector must be fitted before updating scores.")
        
        d_t = float(overall_risk_score) - self.mu_0
        self.cusum_upper = float(max(0.0, self.cusum_upper + d_t - self.k))
        self.cusum_lower = float(max(0.0, self.cusum_lower - d_t - self.k))
        
        return {
            "cusum_upper": self.cusum_upper,
            "cusum_lower": self.cusum_lower,
            "cusum_alert_upper": bool(self.cusum_upper > self.h),
            "cusum_alert_lower": bool(self.cusum_lower > self.h)
        }