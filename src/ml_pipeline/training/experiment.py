from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentRun:
    run_id: str
    task: str
    dataset_hash: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "task": self.task, "dataset_hash": self.dataset_hash, "params": self.params, "metrics": self.metrics}
