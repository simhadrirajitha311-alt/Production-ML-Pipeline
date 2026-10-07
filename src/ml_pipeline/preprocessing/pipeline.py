from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MLPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    random_state: int = 42

    def build(self) -> Pipeline:
        transformers = []
        if self.numeric_columns:
            transformers.append(("numeric", Pipeline([("scaler", StandardScaler())]), self.numeric_columns))
        if self.categorical_columns:
            transformers.append(("categorical", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), self.categorical_columns))
        return Pipeline([("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop"))])

    def fit(self, X: pd.DataFrame):
        pipeline = self.build()
        pipeline.fit(X)
        return pipeline
