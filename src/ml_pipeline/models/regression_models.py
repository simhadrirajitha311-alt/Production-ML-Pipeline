from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def get_regression_models() -> dict[str, object]:
    return {
        "dummy_regressor": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "extra_trees_regressor": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
    }
