from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor


class BaselineModel:
    def __init__(self, kind: str = "classification"):
        self.kind = kind
        if kind == "classification":
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            self.model = DummyRegressor(strategy="mean")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
