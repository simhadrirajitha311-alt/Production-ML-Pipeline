from __future__ import annotations

import joblib


def load_model(path: str):
    return joblib.load(path)
