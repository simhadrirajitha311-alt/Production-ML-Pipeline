from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_categorical_encoder(categories: list[str], *, handle_unknown: str = "ignore") -> Pipeline:
    return Pipeline([
        ("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)),
    ])


def build_numeric_transformer() -> Pipeline:
    return Pipeline([("scaler", StandardScaler())])


def build_column_transformer(X, categorical_columns: list[str], numeric_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(("num", build_numeric_transformer(), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", build_categorical_encoder(categorical_columns), categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")
