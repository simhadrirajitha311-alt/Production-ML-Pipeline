from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    task: str = "classification"
    target: str | None = None
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    primary_metric: str = "f1"
    models_enabled: list[str] = field(default_factory=lambda: ["logistic_regression", "random_forest"])
    mlflow_enabled: bool = True
    optimization_enabled: bool = True
    optimization_trials: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        return cls(
            task=raw.get("task", "classification"),
            target=raw.get("target"),
            random_state=int(raw.get("random_state", 42)),
            test_size=float(raw.get("test_size", 0.2)),
            cv_folds=int(raw.get("cv", {}).get("folds", 5)),
            primary_metric=str(raw.get("evaluation", {}).get("primary_metric", "f1")),
            models_enabled=list(raw.get("models", {}).get("enabled", ["logistic_regression", "random_forest"])),
            mlflow_enabled=bool(raw.get("tracking", {}).get("mlflow", True)),
            optimization_enabled=bool(raw.get("optimization", {}).get("enabled", True)),
            optimization_trials=int(raw.get("optimization", {}).get("trials", 20)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "target": self.target,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "primary_metric": self.primary_metric,
            "models_enabled": self.models_enabled,
            "mlflow_enabled": self.mlflow_enabled,
            "optimization_enabled": self.optimization_enabled,
            "optimization_trials": self.optimization_trials,
        }
