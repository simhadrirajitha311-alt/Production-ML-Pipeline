from __future__ import annotations

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def explain_with_shap(model, X, *, sample_size: int = 100):
    if shap is None:
        return {"status": "shap_not_installed"}
    explainer = shap.Explainer(model, X)
    return explainer(X[:sample_size])
