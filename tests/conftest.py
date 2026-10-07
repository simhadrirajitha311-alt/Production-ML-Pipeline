import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
