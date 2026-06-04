import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

sample_flight = {
    "month": 6,
    "day_of_week": 5,
    "day_of_month": 15,
    "hour": 19,
    "distance": 800.0,
    "distance_group": 4,
    "quarter": 2,
    "origin_airport_id": 12478,
    "dest_airport_id": 11298,
    "airline_encoded": 3,
    "airline_delay_rate": 0.25,
    "origin_delay_rate": 0.28,
    "dest_delay_rate": 0.22,
    "route_delay_rate": 0.30
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_high_risk():
    response = client.post("/predict", json=sample_flight)
    assert response.status_code == 200
    data = response.json()
    assert "is_delayed" in data
    assert "delay_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_low_risk():
    low_risk = {**sample_flight, "hour": 6, "month": 1, 
                "airline_delay_rate": 0.05, "route_delay_rate": 0.05}
    response = client.post("/predict", json=low_risk)
    assert response.status_code == 200
    assert response.json()["delay_probability"] < 0.9

def test_probability_range():
    response = client.post("/predict", json=sample_flight)
    prob = response.json()["delay_probability"]
    assert 0.0 <= prob <= 1.0
    