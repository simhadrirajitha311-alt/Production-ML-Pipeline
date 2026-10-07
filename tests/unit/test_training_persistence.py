from pathlib import Path

import pandas as pd

from ml_pipeline.inference.model_loader import load_model
from ml_pipeline.training.trainer import Trainer
from sklearn.dummy import DummyClassifier


def test_trainer_saves_and_loads_model(tmp_path: Path):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    y = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    save_path = tmp_path / "model.joblib"
    trainer = Trainer(model)
    trainer.save(str(save_path))

    loaded = load_model(str(save_path))
    assert loaded.predict(X).shape[0] == len(X)
