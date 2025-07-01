import pandas as pd
import numpy as np
import pytest
import os
from src.data_processing import load_data, feature_engineer, preprocess_data, split_data

# Define dummy data for testing
@pytest.fixture
def dummy_raw_data():
    """Provides a dummy DataFrame simulating raw transaction data."""
    data = {
        'TransactionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'BatchId': [101, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'AccountId': [1001, 1001, 1002, 1001, 1003, 1002, 1004, 1001, 1003, 1004],
        'SubscriptionId': [1, 1, 2, 1, 3, 2, 4, 1, 3, 4],
        'CustomerId': [2001, 2001, 2002, 2001, 2003, 2002, 2004, 2001, 2003, 2004],
        'CurrencyCode': ['KES', 'KES', 'USD', 'KES', 'KES', 'USD', 'KES', 'KES', 'KES', 'KES'],
        'CountryCode': [254, 254, 1, 254, 254, 1, 254, 254, 254, 254],
        'ProviderId': [1, 2, 1, 1, 3, 2, 1, 2, 3, 1],
        'ProductId': [10, 11, 20, 10, 30, 21, 10, 12, 31, 11],
        'ProductCategory': ['A', 'B', 'C', 'A', 'D', 'C', 'A', 'B', 'D', 'B'],
        'ChannelId': ['Web', 'Android', 'IOS', 'Web', 'Web', 'IOS', 'Android', 'Web', 'Android', 'Web'],
        'Amount': [100.0, 200.0, -50.0, 150.0, 500.0, -25.0, 75.0, 300.0, 120.0, 80.0],
        'Value': [100.0, 200.0, 50.0, 150.0, 500.0, 25.0, 75.0, 300.0, 120.0, 80.0],
        'TransactionStartTime': pd.to_datetime([
            '2024-01-01', '2024-01-05', '2024-01-10', '2024-01-02', '2024-01-15',
            '2024-01-12', '2024-01-03', '2024-01-06', '2024-01-16', '2024-01-04'
        ]),
        'PricingStrategy': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        'FraudResult': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0] # Account 1002 and 1003 have fraud
    }
    return pd.DataFrame(data)

# Test load_data function
def test_load_data_success(tmp_path, dummy_raw_data):
    """Tests if load_data successfully loads a CSV."""
    test_file = tmp_path / "test_data.csv"
    dummy_raw_data.to_csv(test_file, index=False)
    df = load_data(str(test_file))
    assert not df.empty
    assert df.shape == dummy_raw_data.shape
    assert 'TransactionId' in df.columns

def test_load_data_file_not_found():
    """Tests load_data's behavior when file is not found (should create dummy)."""
    df = load_data("non_existent_file.csv")
    assert not df.empty
    assert 'TransactionId' in df.columns
    assert df.shape[0] == 10000 # Dummy data size

# Test feature_engineer function
def test_feature_engineer(dummy_raw_data):
    """Tests if feature_engineer correctly creates customer-level features."""
    customer_df = feature_engineer(dummy_raw_data.copy())

    # Check if expected columns are present
    expected_cols = ['AccountId', 'Recency', 'Frequency', 'Monetary', 'HasFraud',
                     'AvgAmount', 'MaxValue', 'MinAmount', 'StdAmount',
                     'UniqueProductCategories', 'UniqueChannels', 'UniqueProviders',
                     'UniqueCurrencies', 'UniqueCountries',
                     'MostFrequentChannel', 'MostFrequentProductCategory']
    for col in expected_cols:
        assert col in customer_df.columns

    # Check number of unique accounts
    assert customer_df.shape[0] == dummy_raw_data['AccountId'].nunique()

    # Verify RFM calculations for a specific account (e.g., AccountId 1001)
    acc_1001_data = dummy_raw_data[dummy_raw_data['AccountId'] == 1001]
    acc_1001_features = customer_df[customer_df['AccountId'] == 1001].iloc[0]

    # Recency: max_date (Jan 16) - last_txn (Jan 06) = 10 days
    max_date = dummy_raw_data['TransactionStartTime'].max()
    expected_recency_1001 = (max_date - acc_1001_data['TransactionStartTime'].max()).days
    assert acc_1001_features['Recency'] == expected_recency_1001

    # Frequency: 4 transactions for AccountId 1001
    assert acc_1001_features['Frequency'] == 4

    # Monetary: 100 + 200 + 150 + 300 = 750
    assert acc_1001_features['Monetary'] == 750.0

    # HasFraud: Account 1001 has no fraud (all FraudResult=0)
    assert acc_1001_features['HasFraud'] == 0

    # HasFraud: Account 1002 has fraud (one FraudResult=1)
    acc_1002_features = customer_df[customer_df['AccountId'] == 1002].iloc[0]
    assert acc_1002_features['HasFraud'] == 1

    # HasFraud: Account 1003 has fraud (one FraudResult=1)
    acc_1003_features = customer_df[customer_df['AccountId'] == 1003].iloc[0]
    assert acc_1003_features['HasFraud'] == 1

    # Check for NaN values introduced (should be handled by fillna)
    assert not customer_df.isnull().any().any()

# Test preprocess_data function
def test_preprocess_data(dummy_raw_data):
    """Tests if preprocess_data correctly applies one-hot encoding and handles NaNs."""
    customer_df = feature_engineer(dummy_raw_data.copy())
    processed_df = preprocess_data(customer_df.copy())

    # Check if original categorical columns are removed and new OHE columns are added
    assert 'MostFrequentChannel' not in processed_df.columns
    assert 'MostFrequentProductCategory' not in processed_df.columns
    assert 'MostFrequentChannel_Web' in processed_df.columns
    assert 'MostFrequentProductCategory_C' in processed_df.columns

    # Check for NaN values after preprocessing (should be handled by fillna)
    assert not processed_df.isnull().any().any()

    # Check that 'AccountId' is still present (it's an identifier, not a feature for model)
    assert 'AccountId' in processed_df.columns


# Test split_data function
def test_split_data(dummy_raw_data):
    """Tests if split_data correctly splits data and maintains stratification."""
    customer_df = feature_engineer(dummy_raw_data.copy())
    processed_df = preprocess_data(customer_df.copy())

    target_column = 'HasFraud'
    X_train, X_test, y_train, y_test = split_data(processed_df, target_column, test_size=0.5, random_state=42)

    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == processed_df.drop(columns=[target_column, 'AccountId']).shape[1]

    # Check stratification (approximately, due to small sample size)
    # For dummy data with 4 accounts, 2 good, 2 bad, a 50/50 split might be 1 good/1 bad in each
    # For larger datasets, this check is more meaningful.
    assert y_train.value_counts(normalize=True).iloc[0] == pytest.approx(y_test.value_counts(normalize=True).iloc[0], abs=0.1)
