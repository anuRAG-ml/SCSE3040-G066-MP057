import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
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
def predict(data: PredictionInput):
    features = pd.DataFrame([data.model_dump()])
    prediction = float(model.predict(prepare_features(features))[0])

    logger.info(json.dumps({
        "event": "prediction",
        "predicted_metres_per_shift": prediction,
    }))

    return {"prediction": round(prediction, 2)}
