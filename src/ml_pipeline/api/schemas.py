from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    input_data: dict


class TrainRequest(BaseModel):
    dataset_path: str
    target: str
    task: str = "classification"
