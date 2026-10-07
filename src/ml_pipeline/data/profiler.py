from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numerical_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    potential_id_columns: list[str]
    missing_values: dict[str, float]
    duplicate_rate: float
    constant_columns: list[str]
    high_cardinality_columns: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "datetime_features": self.datetime_features,
            "potential_id_columns": self.potential_id_columns,
            "missing_values": self.missing_values,
            "duplicate_rate": self.duplicate_rate,
            "constant_columns": self.constant_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
        }


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> DatasetProfile:
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    numerical = list(df.select_dtypes(include=["number"]).columns)
    categorical = list(df.select_dtypes(exclude=["number", "datetime"]).columns)
    datetime_cols = list(df.select_dtypes(include=["datetime"]).columns)

    if target is not None:
        for col in [target]:
            if col in categorical:
                categorical.remove(col)
            if col in numerical:
                numerical.remove(col)
            if col in datetime_cols:
                datetime_cols.remove(col)

    missing_values = {col: round(float(value), 4) for col, value in df.isna().mean().items() if float(value) > 0}
    duplicate_rate = float(df.duplicated().mean()) if len(df) > 0 else 0.0
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    high_cardinality = {col: int(df[col].nunique(dropna=True)) for col in categorical if df[col].nunique(dropna=True) > 20}

    potential_id = []
    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.9 and df[col].dtype != "object":
            potential_id.append(col)

    return DatasetProfile(
        rows=int(len(df)),
        columns=int(len(df.columns)),
        numerical_features=numerical,
        categorical_features=categorical,
        datetime_features=datetime_cols,
        potential_id_columns=potential_id,
        missing_values=missing_values,
        duplicate_rate=duplicate_rate,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality,
    )
