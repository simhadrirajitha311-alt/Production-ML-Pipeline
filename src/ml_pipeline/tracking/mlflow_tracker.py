from __future__ import annotations

from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


class MLFlowTracker:
    def __init__(self, experiment_name: str = "automl-production-pipeline"):
        self.experiment_name = experiment_name

    def start_run(self) -> Any:
        if mlflow is None:
            return None
        return mlflow.start_run(run_name=self.experiment_name)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if mlflow is not None:
            mlflow.log_metrics(metrics)
