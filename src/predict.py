import pandas as pd
import numpy as np
import joblib
import os

class CreditScoringPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initializes the CreditScoringPredictor by loading the trained model and scaler.

        Args:
            model_path (str): Path to the serialized model (.pkl).
            scaler_path (str): Path to the serialized scaler (.pkl).
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model or scaler file not found. "
                  f"Ensure '{model_path}' and '{scaler_path}' exist.")
            self.model = None
            self.scaler = None
            # For demonstration, create dummy model/scaler if not found
            print("Creating dummy model/scaler for demonstration.")
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            self.model = LogisticRegression()
            self.scaler = StandardScaler()
            # Fit dummy scaler to some dummy data to avoid errors
            self.scaler.fit(np.random.rand(10, 10))

    def _transform_probability_to_score(self, prob_bad: float, A: float = 600, B: float = 20) -> int:
        """
        Transforms a probability of being 'bad' into a credit score.
        Formula: Credit Score = A - B * log(Odds)
        Where Odds = P(Bad) / (1 - P(Bad))

        Args:
            prob_bad (float): The predicted probability of being a 'bad' customer (between 0 and 1).
            A (float): Offset parameter for the score. (e.g., target score for specific odds)
            B (float): Scaling factor (e.g., points to double the odds).

        Returns:
            int: The calculated credit score, rounded to nearest integer.
        """
        if not (0 <= prob_bad <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        if prob_bad == 0: # Handle cases where probability is exactly 0
            odds = 1e-9 / (1 - 1e-9) # Smallest possible odds
        elif prob_bad == 1: # Handle cases where probability is exactly 1
            odds = 1e9 / (1 - 1e9) # Largest possible odds
        else:
            odds = prob_bad / (1 - prob_bad)

        credit_score = A - B * np.log(odds)
        return int(round(credit_score))

    def predict_score_and_terms(self, customer_features: pd.DataFrame) -> dict:
        """
        Predicts the risk probability, assigns a credit score, and recommends loan terms
        for a new customer.

        Args:
            customer_features (pd.DataFrame): A DataFrame containing preprocessed
                                              features for one or more new customers.
                                              Columns must match the training data.

        Returns:
            dict: A dictionary containing:
                - 'risk_probability': Predicted probability of being a 'bad' customer.
                - 'credit_score': Derived credit score.
                - 'recommended_loan_amount': Optimal loan amount.
                - 'recommended_loan_duration': Optimal loan duration.
        """
        if self.model is None or self.scaler is None:
            return {"error": "Model or scaler not loaded. Cannot make predictions."}

        # Ensure feature order matches training data
        # In a real system, you'd have a fixed feature set or a pipeline
        # For this example, assume customer_features already has correct columns
        try:
            scaled_features = self.scaler.transform(customer_features)
        except ValueError as e:
            print(f"Error during scaling: {e}")
            print("Ensure input features match the features used during training.")
            return {"error": f"Feature mismatch: {e}"}

        # Predict probability of being 'bad' (class 1)
        prob_bad = self.model.predict_proba(scaled_features)[:, 1][0] # Assuming single customer for now

        # Convert probability to credit score
        credit_score = self._transform_probability_to_score(prob_bad)

        # Determine optimal loan amount and duration based on credit score (Rule-based)
        recommended_loan_amount = 0
        recommended_loan_duration = "Declined"
        interest_rate = "High"

        if credit_score >= 750:
            recommended_loan_amount = 50000 # KES
            recommended_loan_duration = "6 months"
            interest_rate = "Lowest"
        elif credit_score >= 650:
            recommended_loan_amount = 20000 # KES
            recommended_loan_duration = "3 months"
            interest_rate = "Low"
        elif credit_score >= 550:
            recommended_loan_amount = 5000 # KES
            recommended_loan_duration = "1 month"
            interest_rate = "Moderate"
        elif credit_score >= 450:
            recommended_loan_amount = 1000 # KES
            recommended_loan_duration = "2 weeks"
            interest_rate = "High"
        else:
            recommended_loan_amount = 0 # Declined
            recommended_loan_duration = "Declined"
            interest_rate = "N/A"

        return {
            'risk_probability': float(f"{prob_bad:.4f}"),
            'credit_score': credit_score,
            'recommended_loan_amount': recommended_loan_amount,
            'recommended_loan_duration': recommended_loan_duration,
            'interest_rate_tier': interest_rate
        }


if __name__ == "__main__":
    # Example usage:
    MODEL_DIR = '../../models/'
    MODEL_PATH = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

    # Create dummy model/scaler files if they don't exist for local testing
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model/scaler files not found. Running a dummy train to create them.")
        # This is a simplified way to ensure files exist for testing predict.py
        # In a real setup, train.py would be run as part of a proper pipeline.
        from src.data_processing import load_data, feature_engineer, preprocess_data, split_data
        from src.train import train_model, save_model

        raw_df = load_data('../../data/raw/data.csv')
        customer_features_df = feature_engineer(raw_df.copy())
        processed_df = preprocess_data(customer_features_df.copy())
        X, y = processed_df.drop(columns=['HasFraud', 'AccountId']), processed_df['HasFraud']
        X_train, X_test, y_train, y_test = split_data(processed_df, 'HasFraud') # Re-split for consistency
        dummy_model, dummy_scaler = train_model(X_train, y_train, model_type='logistic_regression')
        save_model(dummy_model, dummy_scaler, MODEL_DIR)
        print("Dummy model/scaler created.")

    predictor = CreditScoringPredictor(MODEL_PATH, SCALER_PATH)

    # Example of new customer features (this would come from real-time transaction data)
    # This dummy data needs to match the structure of processed features from data_processing.py
    # For a real prediction, you would use data_processing.feature_engineer and preprocess_data
    # on new incoming transaction data to generate these features.
    dummy_customer_features = pd.DataFrame({
        'Recency': [10],
        'Frequency': [50],
        'Monetary': [15000],
        'TotalFraudTransactions': [0],
        'AvgAmount': [300],
        'MaxValue': [1000],
        'MinAmount': [10],
        'StdAmount': [200],
        'UniqueProductCategories': [5],
        'UniqueChannels': [2],
        'UniqueProviders': [3],
        'UniqueCurrencies': [1],
        'UniqueCountries': [1],
        'MostFrequentChannel_Web': [1], # Example of one-hot encoded feature
        'MostFrequentChannel_Android': [0],
        'MostFrequentChannel_IOS': [0],
        'MostFrequentChannel_PayLater': [0],
        'MostFrequentProductCategory_Electronics': [0],
        'MostFrequentProductCategory_Fashion': [0],
        'MostFrequentProductCategory_Groceries': [1],
        'MostFrequentProductCategory_Services': [0],
    })

    # Ensure the columns match the training data's columns
    # This is a critical step for deployment. In a real system, you'd save the column order.
    # For this example, we'll try to align with the dummy data created in data_processing.py
    # and train.py.
    # Get column names from the scaler's fitted data (if available) or from X_train
    # (assuming X_train from data_processing.py is representative)
    try:
        # Load X_train to get column order
        X_train_cols = pd.read_csv('../../data/processed/X_train.csv').columns.tolist()
        # Align dummy features to this order, filling missing with 0
        aligned_dummy_features = pd.DataFrame(columns=X_train_cols)
        for col in X_train_cols:
            if col in dummy_customer_features.columns:
                aligned_dummy_features[col] = dummy_customer_features[col]
            else:
                aligned_dummy_features[col] = 0 # Fill with 0 for missing OHE columns

        prediction_result = predictor.predict_score_and_terms(aligned_dummy_features)
        print("\nPrediction Result for Dummy Customer:")
        print(prediction_result)

    except FileNotFoundError:
        print("Could not load X_train.csv to align columns. Using dummy features as is.")
        print("This might lead to errors if column order/presence is critical for the model/scaler.")
        prediction_result = predictor.predict_score_and_terms(dummy_customer_features)
        print("\nPrediction Result for Dummy Customer (using unaligned features):")
        print(prediction_result)

