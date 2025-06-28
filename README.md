# KAIM_W05_Credit_Risk-_Probability_Model
This model leverages the eCommerce platform's data, particularly focusing on Recency, Frequency, and Monetary (RFM) patterns, to assess customer creditworthiness. The project will be executed in several phases, culminating in a product that provides risk probability scores, credit scores, and optimal loan recommendations.

# Credit Risk Model for Buy-Now-Pay-Later Service
This repository contains the code and documentation for the Credit Scoring Model developed for Bati Bank's new buy-now-pay-later (BNPL) service. The project aims to assess customer creditworthiness, assign risk probabilities and credit scores, and recommend optimal loan amounts and durations based on eCommerce transaction data.

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # Raw and processed data (added to .gitignore)
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile                 # For building the Docker image
├── docker-compose.yml         # For local orchestration
├── requirements.txt           # Python dependencies
├── .gitignore                 # Specifies files/folders to ignore
└── README.md                  # Project overview and documentation
```
# Credit Scoring Business Understanding
Understanding the foundational concepts of credit risk and regulatory frameworks like Basel II is paramount for developing a robust and responsible credit scoring model.

## How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord, particularly its Pillar 1 (Minimum Capital Requirements) and Pillar 2 (Supervisory Review Process), profoundly influences the design of our credit risk model. Basel II mandates that banks hold sufficient capital to cover their credit risk exposures, which requires accurate and reliable measurement of risk components like Probability of Default (PD), Loss Given Default (LGD), and Exposure At Default (EAD).

This emphasis necessitates an interpretable and well-documented model for several critical reasons:

### Regulatory Compliance: 
Regulators require transparency into how risk parameters are derived. A "black box" model that cannot explain its predictions is unlikely to receive approval, as supervisors need to validate the model's assumptions, methodology, and performance.

### Risk Management: 
Bank management and risk officers need to understand why a customer is assigned a particular risk score. This understanding is crucial for setting appropriate loan terms, managing portfolio risk, and making informed strategic decisions. Interpretability allows for effective challenge and refinement of the model.

### Model Validation & Auditability: 
The model must be continuously validated internally and externally. A well-documented model facilitates this process by providing clear explanations of data sources, feature engineering, algorithms, and evaluation metrics, making it easier to audit and reproduce results.

### Business Strategy & Communication: 
For a new service like BNPL, the model's outputs (risk probability, credit score) need to be clearly understood by business development, sales, and customer service teams to effectively communicate with customers and align with Bati Bank's overall lending strategy.

## Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
For a new buy-now-pay-later service, historical "default" data (i.e., customers failing to repay BNPL loans) does not yet exist. Therefore, creating a proxy variable becomes absolutely necessary to enable model training. This proxy acts as a substitute for the true default event, allowing us to build a predictive model based on available historical customer behavior. In this project, we are leveraging the FraudResult from the eCommerce platform as a primary proxy for high-risk behavior, assuming that fraudulent transactions indicate a similar lack of financial reliability as a loan default.

However, relying on a proxy variable carries several potential business risks:

### Proxy Mismatch: 
The most significant risk is that the proxy variable (FraudResult) may not perfectly capture the nuances of actual credit default. A customer engaging in fraud might behave differently than a customer who genuinely struggles to repay a loan due to financial hardship. This mismatch can lead to:

False Positives (Type I Error): Classifying "good" customers (who would have repaid) as "bad" due to their historical fraudulent transactions. This leads to missed revenue opportunities and potentially alienates valuable customers.

False Negatives (Type II Error): Classifying "bad" customers (who would default) as "good" because their past fraudulent behavior wasn't detected or didn't fit the FraudResult definition. This results in actual loan losses for Bati Bank.

### Limited Scope: 
The FraudResult only captures one type of problematic behavior. It may not account for other factors that lead to default, such as sudden job loss, unexpected expenses, or poor financial planning, which are not reflected in transaction fraud data.

### Data Quality & Bias: 
The quality and completeness of the FraudResult data from the eCommerce platform are critical. Any biases or inaccuracies in how fraud was historically detected and labeled will be propagated into our credit scoring model.

### Model Drift: 
As the BNPL service matures and actual default data becomes available, the relationship between the FraudResult proxy and true default may change, leading to model degradation over time. Continuous monitoring and eventual retraining with actual default data will be essential.

## What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
In a regulated financial context, the choice between model complexity and interpretability involves significant trade-offs:

## Simple, Interpretable Models (e.g., Logistic Regression, especially with Weight of Evidence - WoE):

### Pros:

High Interpretability: Easy to understand how each feature influences the prediction. This is crucial for regulatory approval (Basel II Pillar 2), model validation, and explaining decisions to customers.

Transparency: Allows for clear audit trails and understanding of model logic.

Robustness: Often less prone to overfitting on smaller datasets.

Computational Efficiency: Faster to train and predict.

Regulatory Preference: Historically favored by regulators due to their transparency.

### Cons:

Lower Predictive Power: May not capture complex, non-linear relationships in the data as effectively as more complex models, potentially leading to lower accuracy and higher expected losses.

Feature Engineering Intensive: Often requires more manual feature engineering (e.g., WoE transformation) to capture non-linearities, which can be time-consuming.

## Complex, High-Performance Models (e.g., Gradient Boosting - XGBoost, LightGBM):

### Pros:

Higher Predictive Power: Often achieve superior accuracy by capturing intricate patterns and non-linear interactions in the data, potentially leading to lower actual default rates and better capital allocation.

Automated Feature Interaction: Can automatically discover complex relationships between features without extensive manual engineering.

Handles Diverse Data: Can naturally handle various data types and missing values.

### Cons:

Lower Interpretability ("Black Box"): It's challenging to explain why a specific prediction was made, making it difficult to satisfy regulatory requirements and build trust.

Complexity in Validation: More difficult to validate and audit, requiring advanced techniques like SHAP or LIME for post-hoc interpretability.

Overfitting Risk: More prone to overfitting if not properly regularized or validated, especially with limited data.

Computational Cost: Can be more resource-intensive to train and tune.

## Trade-off Conclusion for Bati Bank:

For Bati Bank's initial BNPL service, especially under the scrutiny of Basel II, a hybrid approach or a phased strategy might be optimal:

Initial Phase: Start with a Logistic Regression model. It offers immediate interpretability and regulatory comfort, allowing Bati Bank to quickly launch the service with a transparent risk assessment. The FraudResult proxy, being a clear binary, lends itself well to this.

Subsequent Phases: As more actual BNPL repayment data becomes available and the model matures, explore Gradient Boosting models. Implement advanced interpretability techniques (e.g., SHAP values) to gain insights into their predictions. This allows Bati Bank to leverage the higher predictive power for better risk differentiation while still striving for explainability to meet regulatory demands.

The key is to balance predictive accuracy with the imperative for transparency and auditability in a highly regulated financial environment.
