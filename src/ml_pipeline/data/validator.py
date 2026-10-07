from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "errors": self.errors, "warnings": self.warnings, "details": self.details}


def validate_dataframe(df: pd.DataFrame, target: str | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    if df is None or df.empty:
        errors.append("Dataset is empty.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, details=details)

    if df.duplicated().any():
        dup_rate = float(df.duplicated().mean())
        warnings.append(f"Duplicate rows detected: {dup_rate:.2%}.")
        details["duplicate_rate"] = dup_rate

    if target is not None and target not in df.columns:
        errors.append(f"Target column '{target}' is missing from the dataset.")

    if target is not None and target in df.columns and df[target].isna().all():
        errors.append(f"Target column '{target}' contains only missing values.")

    bad_numeric = df.select_dtypes(include=[np.number]).columns[
        df.select_dtypes(include=[np.number]).apply(lambda s: np.isinf(s).any())
    ]
    if len(bad_numeric) > 0:
        errors.append(f"Infinite values detected in numeric columns: {list(bad_numeric)}")

    if target is not None and target in df.columns and df[target].nunique(dropna=True) <= 1:
        warnings.append(f"Target '{target}' has fewer than 2 unique values; classification may be invalid.")

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if constant_cols:
        warnings.append(f"Constant columns detected: {constant_cols[:10]}")
    details["constant_columns"] = constant_cols

    is_valid = not errors
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, details=details)
