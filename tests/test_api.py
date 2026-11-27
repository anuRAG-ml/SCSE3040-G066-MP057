import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "loom_age_yr": 20,
    "thread_count": 40,
    "power_cuts_min": 5,
    "weaver_exp_yr": 8,
    "humidity_pct": 60,
    "shift_hours": 8,
    "fabric": "cotton",
    "shed": "Shed 1",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_invalid_loom_age():
    payload = VALID_PAYLOAD.copy()
    payload["loom_age_yr"] = 51

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_power_cuts():
    payload = VALID_PAYLOAD.copy()
    payload["power_cuts_min"] = -1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
