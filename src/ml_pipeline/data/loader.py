from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Reusable dataset loader for common tabular formats."""

    @staticmethod
    def load(path: str | Path, *, file_format: str | None = None) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        fmt = (file_format or file_path.suffix.lower().lstrip(".")).lower()
        if fmt == "csv":
            df = pd.read_csv(file_path)
        elif fmt == "parquet":
            df = pd.read_parquet(file_path)
        elif fmt in {"xlsx", "xls"}:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {fmt}. Expected csv, parquet, xlsx, or xls.")

        if df.empty:
            raise ValueError(f"Loaded dataset is empty: {file_path}")
        return df

    @staticmethod
    def load_many(paths: list[str | Path]) -> list[pd.DataFrame]:
        return [DataLoader.load(path) for path in paths]
