from __future__ import annotations

from sklearn.metrics import davies_bouldin_score, silhouette_score


def clustering_metrics(X, labels) -> dict[str, float]:
    metrics = {"silhouette": float(silhouette_score(X, labels))}
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except ValueError:
        metrics["davies_bouldin"] = float("nan")
    return metrics
