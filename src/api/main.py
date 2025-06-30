from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse
import joblib
import numpy as np

app = FastAPI(title="Credit Risk API")
model = joblib.load("models/best_model.pkl")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = np.array([[  # exactly in the same 15‑col order
        request.BatchId,
        request.AccountId,
        request.SubscriptionId,
        request.CurrencyCode,
        request.ProviderId,
        request.ProductId,
        request.ProductCategory,
        request.ChannelId,
        request.PricingStrategy,
        request.Amount,
        request.Value,
        request.hour,
        request.day,
        request.month,
        request.year
    ]])
    proba = model.predict_proba(features)[0, 1]
    return PredictResponse(
        risk_probability=proba,
        is_high_risk=int(proba >= 0.5)
    )
