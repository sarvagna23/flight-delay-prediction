from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from database import init_db, log_prediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'flight_model.pkl')

app = FastAPI(title="Flight Delay Prediction API", version="1.0.0")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

init_db()

class FlightRequest(BaseModel):
    month: int
    day_of_week: int
    day_of_month: int
    hour: int
    distance: float
    distance_group: int
    quarter: int
    origin_airport_id: int
    dest_airport_id: int
    airline_encoded: int
    airline_delay_rate: float
    origin_delay_rate: float
    dest_delay_rate: float
    route_delay_rate: float

class PredictionResponse(BaseModel):
    is_delayed: bool
    delay_probability: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "Flight Delay Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightRequest):
    try:
        is_weekend = 1 if flight.day_of_week in [6, 7] else 0
        is_evening = 1 if flight.hour >= 18 else 0
        is_morning = 1 if flight.hour <= 9 else 0
        is_peak_hour = 1 if flight.hour in [7, 8, 17, 18, 19] else 0

        features = np.array([[
            flight.month, flight.day_of_week, flight.day_of_month, flight.hour,
            flight.distance, flight.distance_group, flight.quarter,
            is_weekend, is_evening, is_morning, is_peak_hour,
            flight.origin_airport_id, flight.dest_airport_id, flight.airline_encoded,
            flight.airline_delay_rate, flight.origin_delay_rate,
            flight.dest_delay_rate, flight.route_delay_rate
        ]])

        prob = model.predict_proba(features)[0][1]
        is_delayed = bool(prob >= 0.5)
        risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

        log_prediction(
            flight.month, flight.day_of_week, flight.hour,
            str(flight.airline_encoded), str(flight.origin_airport_id),
            str(flight.dest_airport_id), flight.distance, int(is_delayed), float(prob)
        )

        return PredictionResponse(
            is_delayed=is_delayed,
            delay_probability=round(float(prob), 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)