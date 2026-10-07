from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans


def get_clustering_models() -> dict[str, object]:
    return {
        "kmeans": KMeans(n_clusters=3, random_state=42, n_init=10),
        "agglomerative": AgglomerativeClustering(n_clusters=3),
        "dbscan": DBSCAN(eps=0.5, min_samples=5),
    }
