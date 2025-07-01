from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Dict, Union

# Import functions from data_processing and predict modules
from src.data_processing import feature_engineer, preprocess_data
from src.predict import CreditScoringPredictor

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="API for assessing creditworthiness for Buy-Now-Pay-Later service.",
    version="1.0.0"
)

# --- Load Model and Scaler ---
# Define paths relative to the project root
MODEL_DIR = './models/' # When running via Docker, current dir is project root
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Initialize the predictor globally to load model/scaler once
predictor = CreditScoringPredictor(MODEL_PATH, SCALER_PATH)

# --- Pydantic Models for Request and Response ---
class Transaction(BaseModel):
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
    TransactionStartTime: str # Will be parsed to datetime
    PricingStrategy: int
    # FraudResult is not expected for new transactions, but might be present in historical data
    # For API input, we assume it's not provided or is 0 for new transactions.
    FraudResult: int = 0 # Default to 0 for new transactions

class CreditScoreResponse(BaseModel):
    AccountId: int
    risk_probability: float
    credit_score: int
    recommended_loan_amount: float
    recommended_loan_duration: str
    interest_rate_tier: str
    message: str = "Prediction successful."

class ErrorResponse(BaseModel):
    detail: str

# --- API Endpoints ---

@app.get("/")
async def root():
    """
    Root endpoint for the Credit Scoring API.
    """
    return {"message": "Welcome to Bati Bank Credit Scoring API! Visit /docs for API documentation."}

@app.post("/predict_credit_score/", response_model=Union[List[CreditScoreResponse], ErrorResponse])
async def predict_credit_score(transactions: List[Transaction]):
    """
    Predicts credit score and recommends loan terms for a list of new customer transactions.

    Expects a list of transaction objects for one or more customers.
    The model will aggregate these transactions to derive customer-level features.
    """
    if not predictor.model or not predictor.scaler:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Cannot process request.")

    # Convert list of Transaction Pydantic models to pandas DataFrame
    transactions_df = pd.DataFrame([t.dict() for t in transactions])

    # Ensure 'TransactionStartTime' is datetime
    transactions_df['TransactionStartTime'] = pd.to_datetime(transactions_df['TransactionStartTime'])

    # Get unique AccountIds to process each customer
    customer_ids = transactions_df['AccountId'].unique()
    results = []

    # In a real-world scenario, you'd likely aggregate all transactions for a given customer
    # over a defined look-back period (e.g., last 3 months) to generate features.
    # For this example, we'll process each unique AccountId present in the input transactions.
    # This assumes the input `transactions` list contains all relevant transactions for the
    # customers you want to score at this moment.

    for account_id in customer_ids:
        customer_txns = transactions_df[transactions_df['AccountId'] == account_id].copy()

        # 1. Feature Engineering (using data_processing.py logic)
        # Note: For real-time, this needs to be robust to single/few transactions.
        # The `feature_engineer` function currently expects a DataFrame with multiple transactions
        # to calculate things like Frequency, Monetary, etc.
        # For a single transaction, Frequency would be 1, Monetary would be Value, etc.
        # It's crucial that the feature engineering logic here matches `data_processing.py`
        # and what the model was trained on.

        # For simplicity in API, let's assume `feature_engineer` can handle single-customer data.
        # If the model was trained on aggregated customer data, the input to `predict_score_and_terms`
        # must be in the same format.
        try:
            customer_features_df = feature_engineer(customer_txns)
            # Remove 'HasFraud' if it was generated, as it's the target, not an input feature for prediction
            if 'HasFraud' in customer_features_df.columns:
                customer_features_df = customer_features_df.drop(columns=['HasFraud'])
            if 'AccountId' in customer_features_df.columns:
                customer_features_df = customer_features_df.drop(columns=['AccountId'])

            # 2. Preprocessing (one-hot encoding etc., must match training preprocessing)
            # This is a simplified call; in a robust system, the preprocessing pipeline
            # (including fitted encoders/scalers) would be loaded and applied.
            # For now, we'll manually handle categorical features based on what was in data_processing.py
            # and assume the model expects one-hot encoded features.

            # Get the expected columns from the scaler (which was fitted on X_train)
            # A better approach is to save the column names or a preprocessing pipeline
            # during training and load it here.
            try:
                # This assumes X_train.csv was saved by data_processing.py
                X_train_cols = pd.read_csv('./data/processed/X_train.csv').columns.tolist()
            except FileNotFoundError:
                raise HTTPException(status_code=500, detail="X_train.csv not found. Cannot determine expected feature columns. Please ensure training data is processed and saved.")

            # Apply one-hot encoding to the current customer's features
            categorical_cols_to_encode = customer_features_df.select_dtypes(include='object').columns.tolist()
            customer_features_processed = pd.get_dummies(customer_features_df, columns=categorical_cols_to_encode, drop_first=True)

            # Align columns with the model's expected input features (X_train_cols)
            # Add missing columns (from X_train_cols) and fill with 0
            missing_cols = set(X_train_cols) - set(customer_features_processed.columns)
            for c in missing_cols:
                customer_features_processed[c] = 0
            # Ensure the order of columns is the same as during training
            customer_features_aligned = customer_features_processed[X_train_cols]

            # 3. Predict Credit Score and Loan Terms
            prediction = predictor.predict_score_and_terms(customer_features_aligned)

            if "error" in prediction:
                raise HTTPException(status_code=500, detail=prediction["error"])

            results.append(CreditScoreResponse(
                AccountId=account_id,
                risk_probability=prediction['risk_probability'],
                credit_score=prediction['credit_score'],
                recommended_loan_amount=prediction['recommended_loan_amount'],
                recommended_loan_duration=prediction['recommended_loan_duration'],
                interest_rate_tier=prediction['interest_rate_tier']
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing AccountId {account_id}: {str(e)}")

    return results

