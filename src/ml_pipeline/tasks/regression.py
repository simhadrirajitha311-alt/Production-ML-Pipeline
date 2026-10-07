from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_regression_baseline(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    preds = model.predict(X)
    return {
        "model": "DummyRegressor",
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(mean_squared_error(y, preds, squared=False)),
        "r2": float(r2_score(y, preds)),
    }


def evaluate_regression_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        results[name] = {
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": float(mean_squared_error(y, preds, squared=False)),
            "r2": float(r2_score(y, preds)),
        }
    return results
