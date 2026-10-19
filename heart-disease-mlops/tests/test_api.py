import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(model_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(model_path))
        from api.app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert set(body) == {"prediction", "label", "confidence", "probability_disease"}


def test_predict_accepts_missing_optional_fields(client, sample_payload):
    sample_payload.pop("ca")
    sample_payload["thal"] = None
    assert client.post("/predict", json=sample_payload).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("sex", 2), ("cp", 9), ("thal", 5)])
def test_predict_rejects_invalid_values(client, sample_payload, field, value):
    sample_payload[field] = value
    assert client.post("/predict", json=sample_payload).status_code == 422


def test_predict_rejects_missing_required_field(client, sample_payload):
    del sample_payload["chol"]
    assert client.post("/predict", json=sample_payload).status_code == 422


def test_metrics_endpoint_exposes_counters(client, sample_payload):
    client.post("/predict", json=sample_payload)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
    assert "api_request_latency_seconds" in text
