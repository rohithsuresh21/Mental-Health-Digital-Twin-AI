import numpy as np
from typing import Dict, Any, Union, Optional, List
from scipy.linalg import pinvh
from scipy.stats import chi2

class ProductionMahalanobisDetector:

    def __init__(self, regularization: float = 1e-5, min_regularization_frac: float = 1e-3,
                 exclude_dims: Optional[List[int]] = None) -> None:
        self.regularization: float = regularization
        self.min_regularization_frac: float = min_regularization_frac
        self.exclude_dims: List[int] = exclude_dims or []
        self.mean: np.ndarray = np.array([])
        self.pseudo_inverse: np.ndarray = np.array([])
        self.dof: int = 0
        self.is_fitted: bool = False

    def _select(self, X: np.ndarray) -> np.ndarray:
        if not self.exclude_dims:
            return X
        keep = [i for i in range(X.shape[1]) if i not in set(self.exclude_dims)]
        return X[:, keep]

    def fit(self, X: np.ndarray) -> "ProductionMahalanobisDetector":
        X_arr = np.asarray(X, dtype=np.float64)
        X_arr = self._select(X_arr)
        self.mean = np.mean(X_arr, axis=0)

        cov = np.cov(X_arr, rowvar=False)
        if X_arr.shape[0] == 1:
            cov = np.array([[cov]])
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        trace_scale = np.trace(cov) / cov.shape[0]
        reg = max(self.regularization, self.min_regularization_frac * trace_scale)
        cov_reg = cov + np.eye(cov.shape[0]) * reg

        self.pseudo_inverse = pinvh(cov_reg)
        self.dof = cov.shape[0]

        self.is_fitted = True
        return self

    def _compute_raw_distances(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean
        left_product = np.dot(diff, self.pseudo_inverse)
        return np.sqrt(np.maximum(0.0, np.sum(left_product * diff, axis=1)))

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Mahalanobis detector not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float64)
        X_arr = self._select(X_arr)
        raw_distances = self._compute_raw_distances(X_arr)

        squared_distances = raw_distances ** 2
        normalized_scores = chi2.cdf(squared_distances, df=self.dof)
        return np.clip(normalized_scores, 0.0, 1.0)
