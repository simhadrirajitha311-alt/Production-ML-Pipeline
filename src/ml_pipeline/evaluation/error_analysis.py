from __future__ import annotations

from typing import Any


def summarize_errors(y_true, y_pred) -> dict[str, Any]:
    errors = []
    for actual, predicted in zip(y_true, y_pred, strict=False):
        if actual != predicted:
            errors.append({"actual": actual, "predicted": predicted})
    return {"error_count": len(errors), "examples": errors[:10]}
