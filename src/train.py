import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Custom RandomOverSampler implementation
class RandomOverSampler:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Identify minority and majority classes
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            print("Only one class found. No oversampling performed.")
            return X, y

        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        majority_count = class_counts[np.argmax(class_counts)]
        minority_count = class_counts[np.argmin(class_counts)]

        # If minority class is empty, return original data
        if minority_count == 0:
            return X, y

        # Calculate how many samples to generate
        n_samples_to_generate = majority_count - minority_count

        if n_samples_to_generate <= 0:
            return X, y # No oversampling needed if minority is already larger or equal

        # Get indices of minority class samples
        minority_indices = np.where(y == minority_class)[0]
        
        # Randomly sample with replacement from minority class
        np.random.seed(self.random_state)
        resampled_indices = np.random.choice(minority_indices, size=n_samples_to_generate, replace=True)

        # Concatenate original minority samples with generated samples
        X_resampled_minority = X.iloc[resampled_indices]
        y_resampled_minority = y.iloc[resampled_indices]

        X_resampled = pd.concat([X, X_resampled_minority], axis=0)
        y_resampled = pd.concat([y, y_resampled_minority], axis=0)

        # Shuffle the resampled data
        shuffled_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled.iloc[shuffled_indices].reset_index(drop=True)
        y_resampled = y_resampled.iloc[shuffled_indices].reset_index(drop=True)

        return X_resampled, y_resampled


def load_processed_data_splits(data_dir: str, target_column: str):
    """
    Loads processed data splits (X_train, X_test, y_train, y_test).

    Args:
        data_dir (str): Directory where processed data CSVs are stored.
        target_column (str): Name of the target column (for y_train/y_test files).

    Returns:
        tuple: X_train, X_test, y_train, y_test DataFrames/Series.
    """
    try:
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train_processed.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test_processed.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze() # .squeeze() to convert DataFrame to Series
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
        print(f"Processed data splits loaded successfully from {data_dir}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"Error: Processed data files not found in {data_dir}. Please run data_processing.py first.")
        # Create dummy data for demonstration if files are not found
        print("Creating dummy data for training script demonstration.")
        num_samples = 1000
        # Ensure dummy data matches expected processed feature structure (e.g., num__ and cat__ prefixes)
        X_dummy = pd.DataFrame(np.random.rand(num_samples, 20), columns=[f'num__feature_{i}' for i in range(10)] + [f'cat__feature_{i}' for i in range(10)])
        y_dummy = pd.Series(np.random.choice([0, 1], num_samples, p=[0.95, 0.05]), name=target_column)
        X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy)
        return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'logistic_regression'):
    """
    Trains the credit risk model.

    Args:
        X_train (pd.DataFrame): Training features (already preprocessed by data_processing.py).
        y_train (pd.Series): Training target.
        model_type (str): Type of model to train ('logistic_regression' or 'gradient_boosting').

    Returns:
        sklearn.base.BaseEstimator: The trained model.
    """
    print(f"Training {model_type} model...")

    # Handle class imbalance using custom RandomOverSampler
    print("Applying RandomOverSampler for class imbalance...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"Original class distribution: {y_train.value_counts()}")
    print(f"Resampled class distribution: {y_train_resampled.value_counts()}")


    if model_type == 'logistic_regression':
        # Using class_weight='balanced' in LogisticRegression is a good alternative/complement to SMOTE
        # It automatically adjusts weights inversely proportional to class frequencies.
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'gradient_boosting'.")

    model.fit(X_train_resampled, y_train_resampled)
    print(f"{model_type} model training completed.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the test set.

    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test features (already preprocessed).
        y_test (pd.Series): Test target.
    """
    print("Evaluating model performance...")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

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
    # cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='roc_auc')
    # print(f"\nCross-validation ROC AUC scores: {cv_scores}")
    # print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


def save_model(model, model_dir: str = './models/'):
    """
    Saves the trained model.

    Args:
        model: The trained model.
        model_dir (str): Directory to save the model.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'credit_risk_model.pkl')

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    PROCESSED_DATA_DIR = './data/processed/'
    TARGET_COLUMN = 'HasFraud'
    MODEL_DIR = './models/'

    # Load preprocessed data splits
    X_train, X_test, y_train, y_test = load_processed_data_splits(PROCESSED_DATA_DIR, TARGET_COLUMN)

    # Train model
    # Choose 'logistic_regression' or 'gradient_boosting'
    trained_model = train_model(X_train, y_train, model_type='logistic_regression')

    # Evaluate model
    evaluate_model(trained_model, X_test, y_test)

    # Save model
    save_model(trained_model, MODEL_DIR)
