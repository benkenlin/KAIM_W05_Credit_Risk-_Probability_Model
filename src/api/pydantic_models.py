from pydantic import BaseModel
from typing import List, Optional

# This file is used to define the data models for FastAPI requests and responses.
# It helps FastAPI validate incoming data and serialize outgoing data.

class Transaction(BaseModel):
    """
    Represents a single transaction record as expected by the API for input.
    """
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str
    CountryCode: int
    ProviderId: int
    ProductId: int
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str # Use string for datetime, FastAPI will handle parsing
    PricingStrategy: int
    FraudResult: int = 0 # Default to 0 for new transactions, as it's not an input for prediction


class CreditScoreResponse(BaseModel):
    """
    Represents the credit scoring prediction response for a single account.
    """
    AccountId: int
    risk_probability: float
    credit_score: int
    recommended_loan_amount: float
    recommended_loan_duration: str
    interest_rate_tier: str
    message: str = "Prediction successful." # Optional message field


class ErrorResponse(BaseModel):
    """
    Represents an error response from the API.
    """
    detail: str
