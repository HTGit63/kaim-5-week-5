from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint_smoke():
    payload = {
        # fill with valid numeric defaults
        "BatchId": 1.0, "AccountId": 1.0, "SubscriptionId": 1.0,
        "CurrencyCode": 0.0, "ProviderId": 1.0, "ProductId": 1.0,
        "ProductCategory": 0.0, "ChannelId": 1.0, "PricingStrategy": 1.0,
        "Amount": 0.0, "Value": 0.0, "hour": 0, "day": 1, "month": 1, "year": 2020
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data and "is_high_risk" in data
