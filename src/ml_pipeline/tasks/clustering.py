from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_clustering(X: pd.DataFrame, *, k: int = 3) -> dict[str, Any]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return {"n_clusters": k, "silhouette": float(score), "labels": labels.tolist()}
