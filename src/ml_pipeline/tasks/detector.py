from __future__ import annotations

from typing import Any

import pandas as pd


def detect_task(df: pd.DataFrame, target: str | None = None, override: str | None = None) -> dict[str, Any]:
    if override is not None:
        return {"task": override, "confidence": 1.0, "target": target}

    if target is None:
        return {"task": "clustering", "confidence": 0.8, "target": None}

    s = df[target]
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique(dropna=True) <= 10 and s.nunique(dropna=True) > 1:
            return {"task": "classification", "confidence": 0.9, "target": target, "classes": int(s.nunique(dropna=True))}
        return {"task": "regression", "confidence": 0.9, "target": target}

    return {"task": "classification", "confidence": 0.9, "target": target, "classes": int(s.nunique(dropna=True))}
