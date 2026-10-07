import pandas as pd

from ml_pipeline.tasks.detector import detect_task


def test_detect_classification_task():
    df = pd.DataFrame({"feature_1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    result = detect_task(df, target="target")
    assert result["task"] == "classification"


def test_detect_clustering_when_no_target():
    df = pd.DataFrame({"feature_1": [1, 2, 3, 4], "feature_2": [5, 6, 7, 8]})
    result = detect_task(df)
    assert result["task"] == "clustering"
