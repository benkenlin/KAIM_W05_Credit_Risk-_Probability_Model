from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Dict, Union
import joblib # To load the preprocessing pipeline

# Import classes/functions from data_processing and predict modules
from src.data_processing import CustomerAggregator, load_raw_data # We need CustomerAggregator
from src.predict import CreditScoringPredictor

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="API for assessing creditworthiness for Buy-Now-Pay-Later service.",
    version="1.0.0"
)

# --- Load Model, Scaler, and Preprocessing Pipeline ---
# Define paths relative to the project root (which is /app in Docker)
MODEL_DIR = './models/'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
PREPROCESSING_PIPELINE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.pkl')

# Initialize the predictor globally to load model/scaler once
predictor = CreditScoringPredictor(MODEL_PATH, SCALER_PATH)

# Load the preprocessing pipeline
preprocessing_pipeline = None
try:
    preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
    print(f"Preprocessing pipeline loaded successfully from {PREPROCESSING_PIPELINE_PATH}")
except FileNotFoundError:
    print(f"Error: Preprocessing pipeline file not found at {PREPROCESSING_PIPELINE_PATH}. "
          "Please ensure data_processing.py has been run to create it.")
    # In a production environment, you might want to raise an exception or exit here.
    # For now, we'll allow the app to start but predictions will fail.
except Exception as e:
    print(f"Error loading preprocessing pipeline: {e}")


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
    FraudResult: int = 0 # Default to 0 for new transactions, as it's not an input for prediction

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
    if not predictor.model or not predictor.scaler or not preprocessing_pipeline:
        raise HTTPException(status_code=500, detail="Model, scaler, or preprocessing pipeline not loaded. Cannot process request.")

    # Convert list of Transaction Pydantic models to pandas DataFrame
    transactions_df = pd.DataFrame([t.dict() for t in transactions])

    # Ensure 'TransactionStartTime' is datetime
    transactions_df['TransactionStartTime'] = pd.to_datetime(transactions_df['TransactionStartTime'])

    # Get unique AccountIds to process each customer
    customer_ids = transactions_df['AccountId'].unique()
    results = []

    for account_id in customer_ids:
        customer_txns = transactions_df[transactions_df['AccountId'] == account_id].copy()

        try:
            # Step 1: Feature Engineering (Aggregation) using CustomerAggregator
            # This will create customer-level features from the provided transactions
            # Note: CustomerAggregator expects raw transaction data.
            customer_features_aggregated = CustomerAggregator().transform(customer_txns)

            # Drop target and identifier columns from aggregated features before preprocessing
            # These columns should not be passed to the preprocessing pipeline or model
            features_to_drop_from_agg = ['HasFraud', 'AccountId', 'TotalFraudTransactions']
            customer_features_for_preprocessing = customer_features_aggregated.drop(
                columns=[col for col in features_to_drop_from_agg if col in customer_features_aggregated.columns]
            )

            # Step 2: Preprocessing using the loaded pipeline
            # This applies imputation, scaling, and WOE encoding
            # It's crucial that the columns in customer_features_for_preprocessing match
            # the columns the pipeline was fitted on (X_train_agg from data_processing.py)
            processed_features_array = preprocessing_pipeline.transform(customer_features_for_preprocessing)

            # Convert the processed array back to a DataFrame with correct column names
            # This is essential because the predictor expects a DataFrame with named columns
            processed_features_df = pd.DataFrame(
                processed_features_array,
                columns=preprocessing_pipeline.get_feature_names_out()
            )

            # Step 3: Predict Credit Score and Loan Terms
            # The predictor expects a DataFrame, even if it's a single row
            prediction = predictor.predict_score_and_terms(processed_features_df)

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
            # Log the full traceback for debugging in production
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error processing AccountId {account_id}: {str(e)}")

    return results
