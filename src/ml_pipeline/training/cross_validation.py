from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


def cross_validate_model(model, X, y, *, task: str = "classification", cv_folds: int = 5, scoring: str = "f1"):
    if task == "classification":
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = float(model.score(X_val, y_val)) if hasattr(model, "score") else float(np.mean(pred == y_val))
        scores.append(score)

    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}
