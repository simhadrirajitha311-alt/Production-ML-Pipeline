from __future__ import annotations

from __future__ import annotations

from typing import Any

import joblib


class Trainer:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        if self.model is None:
            raise ValueError("A model must be provided before fitting.")
        self.model.fit(X, y)
        return self

    def save(self, path: str, *, compress: int = 3) -> str:
        joblib.dump(self.model, path, compress=compress)
        return path
