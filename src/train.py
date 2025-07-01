import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # For handling class imbalance
from sklearn.preprocessing import StandardScaler # For scaling numerical features

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(file_path: str, target_column: str):
    """
    Loads processed data and separates features (X) from target (y).

    Args:
        file_path (str): Path to the processed CSV file.
        target_column (str): Name of the target column.

    Returns:
        tuple: X (features DataFrame), y (target Series)
    """
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Processed data loaded from {file_path}")
        return X, y
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {file_path}. Please run data_processing.py first.")
        # Create dummy data for demonstration if file is not found
        print("Creating dummy data for training script demonstration.")
        num_samples = 1000
        X_dummy = pd.DataFrame(np.random.rand(num_samples, 10), columns=[f'feature_{i}' for i in range(10)])
        X_dummy['MostFrequentChannel_Web'] = np.random.randint(0, 2, num_samples) # Example OHE feature
        y_dummy = pd.Series(np.random.choice([0, 1], num_samples, p=[0.95, 0.05]), name=target_column)
        return X_dummy, y_dummy


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'logistic_regression'):
    """
    Trains the credit risk model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model to train ('logistic_regression' or 'gradient_boosting').

    Returns:
        sklearn.base.BaseEstimator: The trained model.
        sklearn.preprocessing.StandardScaler: The fitted scaler.
    """
    print(f"Training {model_type} model...")

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Handle class imbalance using SMOTE
    # Only apply SMOTE if there's a minority class to oversample
    if y_train.value_counts()[1] > 0 and y_train.value_counts()[0] > 0:
        print("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"Original class distribution: {y_train.value_counts()}")
        print(f"Resampled class distribution: {y_train_resampled.value_counts()}")
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
        print("SMOTE not applied (no imbalance or only one class).")


    if model_type == 'logistic_regression':
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # 'balanced' can also help
    elif model_type == 'gradient_boosting':
        # Using a simple GradientBoostingClassifier from sklearn for demonstration
        # In a real scenario, consider XGBoost or LightGBM for better performance
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'gradient_boosting'.")

    model.fit(X_train_resampled, y_train_resampled)
    print(f"{model_type} model training completed.")
    return model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the test set.

    Args:
        model: The trained model.
        scaler: The fitted scaler used for preprocessing.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    print("Evaluating model performance...")

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation (optional, but good for robust evaluation)
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=cv, scoring='roc_auc')
    # print(f"\nCross-validation ROC AUC scores: {cv_scores}")
    # print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


def save_model(model, scaler, model_dir: str = '../../models/'):
    """
    Saves the trained model and scaler.

    Args:
        model: The trained model.
        scaler: The fitted scaler.
        model_dir (str): Directory to save the model.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'credit_risk_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = '../../data/processed/customer_features.csv'
    TARGET_COLUMN = 'HasFraud'
    MODEL_DIR = '../../models/'

    # Load data
    X, y = load_processed_data(PROCESSED_DATA_PATH, TARGET_COLUMN)

    # Split data (re-split here to ensure consistency with training script's logic)
    # In a real MLOps pipeline, X_train, X_test, y_train, y_test might be saved directly
    # from data_processing.py and loaded here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    # Choose 'logistic_regression' or 'gradient_boosting'
    trained_model, fitted_scaler = train_model(X_train, y_train, model_type='logistic_regression')

    # Evaluate model
    evaluate_model(trained_model, fitted_scaler, X_test, y_test)

    # Save model and scaler
    save_model(trained_model, fitted_scaler, MODEL_DIR)
