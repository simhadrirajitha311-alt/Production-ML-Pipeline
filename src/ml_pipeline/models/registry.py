from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge


class ModelRegistry:
    def __init__(self) -> None:
        self.registry = {
            "dummy_classifier": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "dummy_regressor": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=42),
            "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "extra_trees_regressor": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
        }

    def get(self, name: str) -> Any:
        if name not in self.registry:
            raise KeyError(f"Model '{name}' is not registered.")
        return self.registry[name]
