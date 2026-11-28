import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.mp057.preprocessing import prepare_features

# JSON logging to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mp057-api")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MP057 Power Loom Output API",
    description="Predicts metres produced per shift.",
    version="1.0.0",
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(json.dumps({
            "event": "http_request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        }))


class PredictionInput(BaseModel):
    loom_age_yr: float = Field(le=50)
    thread_count: float
    power_cuts_min: float = Field(ge=0)
    weaver_exp_yr: float
    humidity_pct: float
    shift_hours: float
    fabric: str
    shed: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput, request: Request):
    features = pd.DataFrame([data.model_dump()])
    prediction = float(model.predict(prepare_features(features))[0])

    logger.info(json.dumps({
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request.state.request_id,
        "predicted_metres_per_shift": prediction,
    }))

    return {"prediction": round(prediction, 2)}
