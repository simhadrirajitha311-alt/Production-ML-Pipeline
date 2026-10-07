from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


def train_classification_baseline(X: pd.DataFrame, y: pd.Series, *, cv: int = 5) -> dict[str, Any]:
    model = DummyClassifier(strategy="most_frequent")
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return {"model": "DummyClassifier", "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())}


def evaluate_classification_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        results[name] = {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="weighted")),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
        }
    return results
