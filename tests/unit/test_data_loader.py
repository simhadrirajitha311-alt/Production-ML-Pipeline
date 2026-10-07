from pathlib import Path

import pandas as pd

from ml_pipeline.data.loader import DataLoader


def test_data_loader_loads_csv(tmp_path: Path):
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "sample.csv"
    data.to_csv(path, index=False)

    out = DataLoader.load(path)

    assert list(out.columns) == ["a", "b"]
    assert out.shape == (2, 2)


def test_data_loader_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    try:
        DataLoader.load(missing)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
