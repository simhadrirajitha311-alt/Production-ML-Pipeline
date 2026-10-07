from __future__ import annotations

from typing import Any

import pandas as pd


class Predictor:
    def __init__(self, model: Any, *, feature_columns: list[str] | None = None):
        self.model = model
        self.feature_columns = feature_columns or []

    def predict(self, input_data: dict[str, Any] | pd.DataFrame):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        if self.feature_columns:
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            df = df[self.feature_columns]
        return self.model.predict(df)
