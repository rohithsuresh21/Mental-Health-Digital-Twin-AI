import numpy as np
from sklearn.neighbors import NearestNeighbors


class NativeKnnDistanceDetector:
    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        self.k: int = k
        self.metric: str = metric
        self.nn_engine_fit = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm='auto',
            metric=self.metric,
            metric_params={"V": np.eye(1)} if metric == "mahalanobis" else None,
            n_jobs=-1
        )
        self.nn_engine_infer = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='auto',
            metric=self.metric,
            metric_params={"V": np.eye(1)} if metric == "mahalanobis" else None,
            n_jobs=-1
        )
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "NativeKnnDistanceDetector":
        X_arr = np.asarray(X, dtype=np.float64)

        if self.metric == "mahalanobis":
            cov = np.cov(X_arr, rowvar=False)
            if X_arr.ndim == 1:
                cov = np.array([[cov]])
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-5
            self.nn_engine_fit.set_params(metric_params={"V": cov_reg})
            self.nn_engine_infer.set_params(metric_params={"V": cov_reg})

        self.nn_engine_fit.fit(X_arr)
        self.nn_engine_infer.fit(X_arr)

        distances, _ = self.nn_engine_fit.kneighbors(X_arr)
        raw_train_scores = np.mean(distances[:, 1:], axis=1)

        self.min_score = float(np.min(raw_train_scores))
        self.max_score = float(np.max(raw_train_scores)) if np.max(raw_train_scores) > self.min_score else self.min_score + 1.0

        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("KNN detector not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float64)

        distances, _ = self.nn_engine_infer.kneighbors(X_arr)
        raw_scores = np.mean(distances, axis=1)

        normalized = (raw_scores - self.min_score) / (self.max_score - self.min_score)
        return np.clip(normalized, 0.0, 1.0)
