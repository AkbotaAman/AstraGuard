from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURES = ["battery", "temperature", "signal", "cpu_load"]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.04, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_fit = False

    def fit(self, df_normal: pd.DataFrame) -> None:
        X = df_normal[FEATURES].to_numpy()
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._is_fit = True

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("AnomalyDetector is not fitted. Call fit() first.")
        X = df[FEATURES].to_numpy()
        Xs = self.scaler.transform(X)
        # IsolationForest: higher = more normal, lower = more anomalous
        raw = self.model.decision_function(Xs)
        # Convert to anomaly score: higher = more anomalous
        anomaly_score = -raw
        return anomaly_score

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """
        Returns boolean array: True where anomaly.
        If threshold not provided, use percentile-based default.
        """
        s = self.score(df)
        if threshold is None:
            # mark top 4% most anomalous by default
            threshold = np.quantile(s, 0.96)
        return s >= threshold
