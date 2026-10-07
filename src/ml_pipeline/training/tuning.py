from __future__ import annotations

from typing import Any

import optuna


class ModelTuner:
    def __init__(self, model_name: str, objective: Any) -> None:
        self.model_name = model_name
        self.objective = objective

    def optimize(self, n_trials: int = 10, random_state: int = 42) -> dict[str, Any]:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=random_state))
        study.optimize(self.objective, n_trials=n_trials)
        return {
            "best_value": float(study.best_value),
            "best_params": study.best_trial.params,
            "study": study,
        }
