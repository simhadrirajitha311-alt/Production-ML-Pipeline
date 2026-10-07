import pandas as pd

from ml_pipeline.data.validator import validate_dataframe


def test_validate_dataframe_detects_empty_dataset():
    df = pd.DataFrame()
    result = validate_dataframe(df, target="target")
    assert result.is_valid is False
    assert any("empty" in message.lower() for message in result.errors)


def test_validate_dataframe_warns_on_constant_column():
    df = pd.DataFrame({"a": [1, 1, 1], "target": [0, 1, 0]})
    result = validate_dataframe(df, target="target")
    assert any("Constant columns" in warning for warning in result.warnings)
