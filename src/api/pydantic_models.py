from pydantic import BaseModel

class PredictRequest(BaseModel):
    BatchId: float
    AccountId: float
    SubscriptionId: float
    CurrencyCode: float
    ProviderId: float
    ProductId: float
    ProductCategory: float
    ChannelId: float
    PricingStrategy: float
    Amount: float
    Value: float
    hour: int
    day: int
    month: int
    year: int

class PredictResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
