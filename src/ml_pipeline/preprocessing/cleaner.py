from __future__ import annotations

import pandas as pd


class DataCleaner:
    def drop_missing_rows(self, df: pd.DataFrame, *, subset: list[str] | None = None) -> pd.DataFrame:
        return df.dropna(subset=subset) if subset else df.dropna()

    def drop_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().copy()

    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, [col for col in df.columns if df[col].nunique(dropna=True) > 1]].copy()
