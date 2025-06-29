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