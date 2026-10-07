from fastapi.testclient import TestClient

from ml_pipeline.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_train_endpoint():
    response = client.post(
        "/train",
        json={"dataset_path": "data/raw/sample.csv", "target": "target", "task": "classification"},
    )
    assert response.status_code in {200, 400}
