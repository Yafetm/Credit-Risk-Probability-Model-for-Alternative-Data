# Credit Risk Probability Model for Alternative Data

This repository contains the implementation of a Credit Scoring Model for Bati Bank, developed as part of the 10 Academy Artificial Intelligence Mastery Week 5 challenge. The model leverages eCommerce transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations for a buy-now-pay-later service.

## Credit Scoring Business Understanding

### 1. Influence of Basel II Accord on Interpretable Models
The Basel II Accord emphasizes robust risk measurement and management to ensure financial stability. It requires banks to maintain adequate capital reserves based on credit risk assessments and to provide transparent, auditable models for regulatory compliance. An interpretable and well-documented model, such as Logistic Regression with Weight of Evidence (WoE), is critical because it allows regulators and stakeholders to understand how risk scores are derived, ensuring compliance with Basel II’s Pillar 1 (minimum capital requirements) and Pillar 2 (supervisory review). Clear documentation also facilitates validation and auditing, reducing the risk of regulatory penalties and enhancing trust in the model’s predictions.

### 2. Necessity and Risks of a Proxy Variable
Since the dataset lacks a direct "default" label, creating a proxy variable (e.g., `is_high_risk` based on RFM metrics) is necessary to categorize customers as high or low risk for model training. This proxy is derived from behavioral data, such as low transaction frequency or monetary value, to approximate default risk. However, using a proxy introduces business risks, including misclassification (e.g., incorrectly labeling a low-risk customer as high-risk), which could lead to lost revenue from denied loans or increased defaults from approving high-risk customers. The proxy’s accuracy depends on the quality of the underlying assumptions, and any misalignment with actual default behavior could undermine the model’s reliability and business outcomes.

### 3. Trade-offs Between Simple and Complex Models
In a regulated financial context, simple models like Logistic Regression with WoE offer high interpretability, making it easier to explain predictions to regulators and stakeholders, aligning with Basel II’s transparency requirements. They are computationally efficient and robust to smaller datasets but may sacrifice predictive power for complex patterns. Conversely, complex models like Gradient Boosting Machines (GBM) often achieve higher predictive accuracy by capturing non-linear relationships but are less interpretable, posing challenges for regulatory audits and explainability. In a regulated environment, the trade-off favors simpler models to ensure compliance and trust, unless the complex model’s performance gains are substantial and can be justified with robust documentation and validation.

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── notebooks/
│   └── 1.0-eda.ipynb         # Exploratory analysis
├── src/
│   ├── init.py
│   ├── data_processing.py    # Feature engineering
│   ├── train.py              # Model training
│   ├── predict.py            # Inference
│   └── api/
│       ├── main.py           # FastAPI application
│       └── pydantic_models.py # Pydantic models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
# Kifiya AIM Week 5 Challenge

This repository contains the solution to the Kifiya AIM Week 5 challenge, focusing on data ingestion, exploratory data analysis, feature engineering, model training, and deployment.

## Task 1: Data Ingestion
- Set up the Git repository and created `.gitignore` to exclude `data/` and other unnecessary files.
- Ingested the dataset into `data/raw/transactions.csv`.
- Committed initial setup files.

## Task 2: Exploratory Data Analysis (EDA)
- Performed EDA in `notebooks/eda.ipynb`, analyzing `transactions.csv` for insights.
- Updated from initial `1.0-eda.ipynb` and committed the final version.

## Task 3: Feature Engineering
- Implemented a pipeline in `src/data_processing.py` to aggregate RFM (Recency, Frequency, Monetary) features.
- Output saved to `data/processed/features.csv` with shape (3742, 4).
- Committed the script and output file.

## Task 4: Model Training
- Trained a RandomForestClassifier using RFM features in `src/train.py`.
- Model saved to `models/rf_model.pkl`.
- Committed the script and model file.

## Task 5: Model Evaluation and Deployment
- Evaluated model performance in `notebooks/2.0-model_evaluation.ipynb` with confusion matrix and classification report.
- Deployed API in `src/api/main.py` for real-time predictions, accessible at `http://127.0.0.1:8000/predict`.
- Tested successfully with `curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Recency\": 10, \"Frequency\": 5, \"Monetary\": 5000}"` returning `{"credit_risk_probability": 0}`.
- Committed the notebook and API file.

## Dependencies
- Python 3.10
- Required packages: `pandas`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn`
- Install via: `pip install pandas scikit-learn joblib fastapi uvicorn`

## How to Run
1. Clone the repository: `git clone <repository-url>`
2. Navigate to the workspace: `cd C:\Users\hp\Desktop\Kifiya AIM\week 5\Technical Content\workspace`
3. Install dependencies: `pip install -r requirements.txt` (create this file if not present with the listed packages)
4. Run data processing: `python src/data_processing.py`
5. Train the model: `python src/train.py`
6. Evaluate the model: `jupyter notebook notebooks/2.0-model_evaluation.ipynb`
7. Start the API: `uvicorn src.api.main:app --reload`
8. Test the API: `curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Recency\": 10, \"Frequency\": 5, \"Monetary\": 5000}"`

## Notes
- The dataset is imbalanced, with only 16 fraud cases out of 3742, affecting minority class performance.
- A scikit-learn version mismatch (1.7.0 vs. 1.5.1) was noted but did not impact results significantly.
