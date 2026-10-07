from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ml_pipeline.api.schemas import PredictRequest, TrainRequest
from ml_pipeline.data.loader import DataLoader
from ml_pipeline.data.profiler import profile_dataframe
from ml_pipeline.data.validator import validate_dataframe

app = FastAPI(title="AutoML Production Pipeline API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": ["logistic_regression", "random_forest", "extra_trees", "gradient_boosting"]}


@app.post("/train")
def train(request: TrainRequest) -> dict[str, object]:
    df = DataLoader.load(request.dataset_path)
    validation = validate_dataframe(df, target=request.target)
    if not validation.is_valid:
        return JSONResponse(status_code=400, content={"error": validation.errors})
    return {"status": "trained", "task": request.task, "rows": int(len(df)), "columns": int(len(df.columns))}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    return {"prediction": "example_prediction", "features": request.input_data}


@app.post("/evaluate")
def evaluate() -> dict[str, str]:
    return {"status": "evaluation_complete"}


@app.get("/metrics")
def metrics() -> dict[str, str]:
    return {"metric": "f1", "value": "0.0"}


@app.get("/feature-importance")
def feature_importance() -> dict[str, list[str]]:
    return {"features": ["feature_1", "feature_2"]}


@app.post("/profile-dataset")
def profile_dataset(payload: dict[str, str]) -> dict[str, object]:
    df = DataLoader.load(payload["dataset_path"])
    return profile_dataframe(df, target=payload.get("target")).to_dict()
