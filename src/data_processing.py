import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
import joblib # For saving the pipeline


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Custom Weight of Evidence (WOE) Transformer.
    Calculates WOE for categorical features based on the target variable.
    Handles missing values by treating them as a separate category.
    """
    def __init__(self, woe_columns=None, smooth_factor=0.001):
        self.woe_columns = woe_columns
        self.woe_maps = {}
        self.smooth_factor = smooth_factor # Smoothing factor to avoid division by zero
        self.fitted_ = False

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Ensure X is a DataFrame for column operations

        if self.woe_columns is None:
            # If not specified, identify all object (categorical) columns
            self.woe_columns = X.select_dtypes(include='object').columns.tolist()
            # Also include numerical columns that might be treated as categorical (e.g., small unique values)
            for col in X.select_dtypes(include=np.number).columns:
                if X[col].nunique() < 50 and X[col].dtype != 'float': # Heuristic for low cardinality numericals
                    self.woe_columns.append(col)
            self.woe_columns = list(set(self.woe_columns)) # Remove duplicates

        for col in self.woe_columns:
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found in X during WOETransformer fit. Skipping.")
                continue

            # Create a temporary DataFrame for WOE calculation, including target
            temp_df = pd.DataFrame({col: X[col], 'target': y})

            # Handle missing values by treating them as a distinct category
            temp_df[col] = temp_df[col].fillna('__MISSING__')

            # Calculate good and bad counts for each category
            # 'good' are target=0, 'bad' are target=1 (assuming 0=good, 1=bad)
            counts = temp_df.groupby(col)['target'].agg(['count', 'sum']).rename(columns={'sum': 'bad'})
            counts['good'] = counts['count'] - counts['bad']

            # Calculate total good and bad counts for the column
            total_good = counts['good'].sum()
            total_bad = counts['bad'].sum()

            # Apply smoothing to avoid division by zero
            counts['prop_good'] = (counts['good'] + self.smooth_factor) / (total_good + 2 * self.smooth_factor)
            counts['prop_bad'] = (counts['bad'] + self.smooth_factor) / (total_bad + 2 * self.smooth_factor)

            # Calculate WOE
            counts['woe'] = np.log(counts['prop_good'] / counts['prop_bad'])

            # Store the WOE mapping
            self.woe_maps[col] = counts['woe'].to_dict()

        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("WOETransformer is not fitted yet. Call 'fit' before 'transform'.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Ensure X is a DataFrame for column operations

        X_transformed = X.copy()
        for col in self.woe_columns:
            if col in X_transformed.columns:
                # Replace missing values with a placeholder before mapping
                X_transformed[col] = X_transformed[col].fillna('__MISSING__')
                # Map categories. If a category is unseen, fill with 0 (or a default WOE, e.g., mean WOE)
                # Using 0 for unseen categories is a common practice.
                X_transformed[col] = X_transformed[col].map(self.woe_maps.get(col, {})).fillna(0)
            else:
                # If a WOE column from fit is not in transform data, add it with 0s
                X_transformed[col] = 0.0
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns output feature names. For WOETransformer, the output feature names
        are the same as the input categorical columns that were transformed.
        """
        if input_features is not None:
            # Filter input_features to only include those that were WOE-transformed
            return [f for f in input_features if f in self.woe_columns]
        return self.woe_columns


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to aggregate transaction-level data to customer-level features
    and define the 'HasFraud' proxy variable.
    """
    def fit(self, X, y=None):
        # No fitting needed for this aggregation step
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw transaction data into customer-level features.

        Args:
            X (pd.DataFrame): Raw transaction DataFrame.

        Returns:
            pd.DataFrame: DataFrame with customer-level features and target variable.
        """
        print("Starting customer-level aggregation and feature extraction...")

        # Ensure TransactionStartTime is datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

        # Extract time-based features from TransactionStartTime at transaction level
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDayOfMonth'] = X['TransactionStartTime'].dt.day
        X['TransactionDayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year

        # Define the 'current date' for Recency calculation as the latest transaction in the dataset
        current_date = X['TransactionStartTime'].max()

        # Aggregate data to customer level (AccountId)
        customer_data = X.groupby('AccountId').agg(
            # RFM features
            Recency=('TransactionStartTime', lambda date: (current_date - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum'), # Total value spent by the customer
            # Fraud proxy: 1 if any transaction for the account was fraudulent, 0 otherwise
            HasFraud=('FraudResult', lambda x: 1 if x.sum() > 0 else 0),
            # Additional numerical aggregates
            AvgAmount=('Value', 'mean'), # Average absolute transaction value
            MaxValue=('Value', 'max'),
            MinAmount=('Value', 'min'),
            StdAmount=('Value', 'std'), # Standard deviation of absolute transaction values
            TotalFraudTransactions=('FraudResult', 'sum'), # Count of fraudulent transactions

            # Count of unique categorical values
            UniqueProductCategories=('ProductCategory', 'nunique'),
            UniqueChannels=('ChannelId', 'nunique'),
            UniqueProviders=('ProviderId', 'nunique'),
            UniqueCurrencies=('CurrencyCode', 'nunique'),
            UniqueCountries=('CountryCode', 'nunique'),

            # Most frequent categorical values (for encoding later)
            MostFrequentChannel=('ChannelId', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            MostFrequentProductCategory=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            MostFrequentCurrencyCode=('CurrencyCode', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            MostFrequentCountryCode=('CountryCode', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            MostFrequentProviderId=('ProviderId', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            MostFrequentPricingStrategy=('PricingStrategy', lambda x: x.mode()[0] if not x.mode().empty else np.nan),

            # Aggregated time-based features
            AvgTransactionHour=('TransactionHour', 'mean'),
            AvgTransactionDayOfMonth=('TransactionDayOfMonth', 'mean'),
            AvgTransactionDayOfWeek=('TransactionDayOfWeek', 'mean'),
            AvgTransactionMonth=('TransactionMonth', 'mean'),
            AvgTransactionYear=('TransactionYear', 'mean'),
            MinTransactionHour=('TransactionHour', 'min'),
            MaxTransactionHour=('TransactionHour', 'max'),
        ).reset_index()

        # Handle potential NaNs from StdAmount for customers with a single transaction
        # and other aggregated features if a customer has no data for a specific type.
        # For StdAmount, NaN means no variance, so it can be 0.
        customer_data['StdAmount'] = customer_data['StdAmount'].fillna(0)

        print("Customer-level aggregation and feature extraction completed.")
        return customer_data


def get_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> Pipeline:
    """
    Creates and returns a scikit-learn Pipeline for preprocessing customer-level features.

    Args:
        numerical_cols (list): List of numerical feature names.
        categorical_cols (list): List of categorical feature names.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    # Preprocessing for numerical features: impute NaNs with median, then standardize
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features: apply Weight of Evidence (WoE) transformation
    # WOETransformer handles missing values by creating a separate bin for NaNs.
    # It's crucial that WOETransformer can handle unseen categories or NaNs gracefully during transform.
    categorical_transformer = Pipeline(steps=[
        ('woe', WOETransformer(woe_columns=categorical_cols))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (like AccountId, if not dropped before)
    )

    print("Preprocessing pipeline created.")
    return preprocessor


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Creating dummy data.")
        # Create dummy data for demonstration if file is not found
        data = {
            'TransactionId': range(10000),
            'BatchId': np.random.randint(1, 50, 10000),
            'AccountId': np.random.randint(100, 5000, 10000),
            'SubscriptionId': np.random.randint(1, 1000, 10000),
            'CustomerId': np.random.randint(100, 5000, 10000),
            'CurrencyCode': np.random.choice(['KES', 'USD', 'UGX'], 10000),
            'CountryCode': np.random.randint(250, 300, 10000),
            'ProviderId': np.random.randint(1, 10, 10000),
            'ProductId': np.random.randint(1000, 1020, 10000),
            'ProductCategory': np.random.choice(['Electronics', 'Groceries', 'Fashion', 'Services'], 10000),
            'ChannelId': np.random.choice(['Web', 'Android', 'IOS', 'PayLater'], 10000),
            'Amount': np.random.uniform(-10000, 50000, 10000),
            'Value': np.abs(np.random.uniform(-10000, 50000, 10000)),
            'TransactionStartTime': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, 10000), unit='D'),
            'PricingStrategy': np.random.randint(0, 5, 10000),
            'FraudResult': np.random.choice([0, 1], 10000, p=[0.995, 0.005]) # Simulate imbalance
        }
        df = pd.DataFrame(data)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        return df


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame (after aggregation and initial feature engineering).
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Splitting data into training and testing sets...")
    # Drop AccountId as it's an identifier, not a feature for the model
    # Also drop TotalFraudTransactions as it's highly correlated with HasFraud (target)
    features_to_drop = [target_column, 'AccountId', 'TotalFraudTransactions']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df[target_column]

    # Stratify by target to maintain class distribution in splits, crucial for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("Data split completed.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    RAW_DATA_PATH = './data/raw/data.csv' # Adjusted path
    PROCESSED_DATA_DIR = './data/processed/' # Adjusted path
    PIPELINE_PATH = './models/preprocessing_pipeline.pkl' # Adjusted path
    TARGET_COLUMN = 'HasFraud'

    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)

    # 1. Load raw data
    raw_df = load_raw_data(RAW_DATA_PATH)

    # 2. Perform customer-level aggregation and initial feature extraction
    customer_features_df = CustomerAggregator().transform(raw_df.copy())

    # Identify numerical and categorical columns for preprocessing pipeline
    # Exclude the target column and AccountId from feature lists
    all_features = [col for col in customer_features_df.columns if col not in [TARGET_COLUMN, 'AccountId', 'TotalFraudTransactions']]
    numerical_cols = customer_features_df[all_features].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = customer_features_df[all_features].select_dtypes(include='object').columns.tolist()

    # 3. Create and fit the preprocessing pipeline
    preprocessing_pipeline = get_preprocessing_pipeline(numerical_cols, categorical_cols)

    # Split data BEFORE fitting the preprocessor to avoid data leakage
    X_agg, y_agg = customer_features_df.drop(columns=[TARGET_COLUMN, 'AccountId', 'TotalFraudTransactions']), customer_features_df[TARGET_COLUMN]
    X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(
        X_agg, y_agg, test_size=0.2, random_state=42, stratify=y_agg
    )

    print("Fitting preprocessing pipeline on training data...")
    # Fit the preprocessing pipeline on the training data
    preprocessing_pipeline.fit(X_train_agg, y_train_agg) # WOETransformer needs y for fitting

    # Transform both training and test data
    print("Transforming training and test data...")
    X_train_processed = pd.DataFrame(
        preprocessing_pipeline.transform(X_train_agg),
        # Call get_feature_names_out AFTER fitting the pipeline
        columns=preprocessing_pipeline.get_feature_names_out()
    )
    X_test_processed = pd.DataFrame(
        preprocessing_pipeline.transform(X_test_agg),
        # Call get_feature_names_out AFTER fitting the pipeline
        columns=preprocessing_pipeline.get_feature_names_out()
    )

    # Save processed data and the fitted pipeline
    X_train_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_processed.csv'), index=False)
    X_test_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_processed.csv'), index=False)
    y_train_agg.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    y_test_agg.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)

    joblib.dump(preprocessing_pipeline, PIPELINE_PATH)
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")
    print(f"Preprocessing pipeline saved to {PIPELINE_PATH}")

    print("\nExample of processed training data head:")
    print(X_train_processed.head())
    print("\nShape of processed training data:", X_train_processed.shape)
