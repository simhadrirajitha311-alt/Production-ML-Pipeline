import pandas as pd

from ml_pipeline.training.baseline import BaselineModel


def test_baseline_classifier_fit_predict():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = pd.Series([1, 0, 1, 0])
    model = BaselineModel("classification").fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
