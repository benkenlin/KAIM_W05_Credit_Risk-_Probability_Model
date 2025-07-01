import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# --- Data Loading ---
# Assuming the main data file is 'data.csv' and is located in the 'data/raw/' directory.
# Xente_Variable_Definitions.csv/.xlsx can be used for understanding column meanings.
try:
    df_train = pd.read_csv('../../data/raw/data.csv')
    print("Data loaded successfully from data.csv.")
except FileNotFoundError:
    print("Data file 'data.csv' not found. Please ensure it is in 'data/raw/'.")
    print("Creating dummy dataframe for demonstration purposes.")
    # Create dummy dataframe for demonstration if file is not found
    data_train = {
        'TransactionId': range(1000),
        'BatchId': np.random.randint(1, 50, 1000),
        'AccountId': np.random.randint(100, 500, 1000),
        'SubscriptionId': np.random.randint(1, 100, 1000),
        'CustomerId': np.random.randint(100, 500, 1000),
        'CurrencyCode': np.random.choice(['KES', 'USD', 'UGX'], 1000),
        'CountryCode': np.random.randint(250, 300, 1000),
        'ProviderId': np.random.randint(1, 10, 1000),
        'ProductId': np.random.randint(1000, 1020, 1000),
        'ProductCategory': np.random.choice(['Electronics', 'Groceries', 'Fashion', 'Services'], 1000),
        'ChannelId': np.random.choice(['Web', 'Android', 'IOS', 'PayLater'], 1000),
        'Amount': np.random.uniform(-10000, 50000, 1000),
        'Value': np.abs(np.random.uniform(-10000, 50000, 1000)),
        'TransactionStartTime': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, 1000), unit='D'),
        'PricingStrategy': np.random.randint(0, 5, 1000),
        'FraudResult': np.random.choice([0, 1], 1000, p=[0.99, 0.01]) # Simulate imbalance
    }
    df_train = pd.DataFrame(data_train)
    df_train['TransactionStartTime'] = pd.to_datetime(df_train['TransactionStartTime'])


# --- 1. Overview of the Data ---
print("\n--- Data Overview (Training Data) ---")
print("Shape:", df_train.shape)
print("\nFirst 5 rows:")
print(df_train.head())
print("\nData types:")
print(df_train.info())

# --- 2. Summary Statistics ---
print("\n--- Summary Statistics (Numerical Features) ---")
print(df_train.describe())

# --- 3. Distribution of Numerical Features ---
print("\n--- Distribution of Numerical Features ---")
numerical_features = df_train.select_dtypes(include=np.number).columns.tolist()
# Exclude IDs and FraudResult for general distribution plots
numerical_features_to_plot = [col for col in numerical_features if 'Id' not in col and col != 'FraudResult']

for col in numerical_features_to_plot:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_train[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Special look at 'Amount' and 'Value' due to their nature (debits/credits vs. absolute)
plt.figure(figsize=(10, 5))
sns.histplot(df_train['Amount'], kde=True, bins=50)
plt.title('Distribution of Amount (Debit/Credit)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_train['Value'], kde=True, bins=50)
plt.title('Distribution of Value (Absolute Amount)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Distribution of FraudResult (our proxy target)
plt.figure(figsize=(6, 4))
sns.countplot(x='FraudResult', data=df_train)
plt.title('Distribution of FraudResult (0: No Fraud, 1: Fraud)')
plt.xlabel('Fraud Result')
plt.ylabel('Count')
plt.show()
fraud_percentage = df_train['FraudResult'].value_counts(normalize=True) * 100
print(f"\nFraudulent transactions (FraudResult=1): {fraud_percentage[1]:.2f}%")
print(f"Non-fraudulent transactions (FraudResult=0): {fraud_percentage[0]:.2f}%")

# --- 4. Distribution of Categorical Features ---
print("\n--- Distribution of Categorical Features ---")
categorical_features = df_train.select_dtypes(include='object').columns.tolist()

for col in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=col, data=df_train, order=df_train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()

# Analyze categorical features in relation to FraudResult
for col in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=col, hue='FraudResult', data=df_train,
                  order=df_train[col].value_counts().index)
    plt.title(f'Distribution of {col} by FraudResult')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.legend(title='FraudResult')
    plt.show()

# --- 5. Correlation Analysis ---
print("\n--- Correlation Analysis (Numerical Features) ---")
# Calculate correlation matrix for numerical features
corr_matrix = df_train[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Correlation with FraudResult specifically
print("\nCorrelation of Numerical Features with FraudResult:")
print(df_train[numerical_features].corr()['FraudResult'].sort_values(ascending=False))

# --- 6. Identifying Missing Values ---
print("\n--- Missing Values Analysis ---")
missing_values = df_train.isnull().sum()
missing_percentage = (df_train.isnull().sum() / len(df_train)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing %': missing_percentage})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))

if missing_df[missing_df['Missing Count'] > 0].empty:
    print("No missing values found in the training dataset.")

# --- 7. Outlier Detection ---
print("\n--- Outlier Detection (Numerical Features) ---")
for col in numerical_features_to_plot:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_train[col])
    plt.title(f'Box Plot of {col} for Outlier Detection')
    plt.xlabel(col)
    plt.show()

# Box plot for Amount and Value
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_train['Amount'])
plt.title('Box Plot of Amount')
plt.xlabel('Amount')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df_train['Value'])
plt.title('Box Plot of Value')
plt.xlabel('Value')
plt.show()

# --- Additional EDA for Credit Risk Modeling ---
# Convert TransactionStartTime to datetime and extract time-based features
df_train['TransactionStartTime'] = pd.to_datetime(df_train['TransactionStartTime'])
df_train['TransactionHour'] = df_train['TransactionStartTime'].dt.hour
df_train['TransactionDayOfWeek'] = df_train['TransactionStartTime'].dt.dayofweek
df_train['TransactionMonth'] = df_train['TransactionStartTime'].dt.month

# Analyze fraud by hour of day
plt.figure(figsize=(12, 6))
sns.countplot(x='TransactionHour', hue='FraudResult', data=df_train)
plt.title('Fraudulent vs. Non-Fraudulent Transactions by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

# Analyze fraud by day of week
plt.figure(figsize=(12, 6))
sns.countplot(x='TransactionDayOfWeek', hue='FraudResult', data=df_train)
plt.title('Fraudulent vs. Non-Fraudulent Transactions by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Count')
plt.show()

# Aggregate data to customer level for RFM and fraud status
# For Recency, we need a 'current date' reference. Let's use the max transaction time in the dataset.
current_date = df_train['TransactionStartTime'].max()

customer_data = df_train.groupby('AccountId').agg(
    Recency=('TransactionStartTime', lambda date: (current_date - date.max()).days),
    Frequency=('TransactionId', 'count'),
    Monetary=('Value', 'sum'),
    TotalFraudTransactions=('FraudResult', 'sum'),
    HasFraud=('FraudResult', lambda x: 1 if x.sum() > 0 else 0), # Our proxy target
    AvgAmount=('Amount', 'mean'),
    MaxValue=('Value', 'max'),
    UniqueProductCategories=('ProductCategory', 'nunique'),
    UniqueChannels=('ChannelId', 'nunique')
).reset_index()

print("\n--- Customer-level Aggregated Data Sample ---")
print(customer_data.head())
print("\nDistribution of HasFraud (Customer Level):")
print(customer_data['HasFraud'].value_counts(normalize=True) * 100)

# Analyze RFM features in relation to HasFraud
rfm_features = ['Recency', 'Frequency', 'Monetary', 'AvgAmount', 'MaxValue', 'UniqueProductCategories', 'UniqueChannels']
for col in rfm_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='HasFraud', y=col, data=customer_data)
    plt.title(f'{col} by HasFraud (Customer Level)')
    plt.xlabel('Has Fraud (0: No, 1: Yes)')
    plt.ylabel(col)
    plt.show()

# Correlation at customer level
customer_corr_matrix = customer_data.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(customer_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Customer-Level Features')
plt.show()

print("\nCorrelation of Customer-Level Features with HasFraud:")
print(customer_data.corr(numeric_only=True)['HasFraud'].sort_values(ascending=False))
