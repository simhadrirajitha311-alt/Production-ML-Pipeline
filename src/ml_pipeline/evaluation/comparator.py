from __future__ import annotations

from typing import Any


def rank_models(results: list[dict[str, Any]], metric: str = "f1") -> list[dict[str, Any]]:
    ranked = sorted(results, key=lambda item: float(item.get(metric, float("-inf"))), reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx
    return ranked


def summarize_model_results(results: list[dict[str, Any]], metric: str = "f1") -> dict[str, Any]:
    return {"metric": metric, "models": rank_models(results, metric=metric)}
