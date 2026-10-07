from __future__ import annotations

from typing import Any


def explain_feature_importance(model, feature_names: list[str]) -> dict[str, Any]:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {name: float(value) for name, value in zip(feature_names, importances, strict=False)}
    return {name: 0.0 for name in feature_names}
