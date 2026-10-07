from __future__ import annotations

from sklearn.inspection import permutation_importance


def permutation_feature_importance(model, X, y, *, n_repeats: int = 5) -> dict[str, float]:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    return {name: float(value) for name, value in zip(X.columns, result.importances_mean, strict=False)}
