# Financial-Fraud-Detection-Using-Machine-Learning-and-Behavioral-Analysis
Overview:
This project investigates fraudulent financial transactions using a structured dataset from Kaggle. By analyzing transactional behavior, merchant patterns, customer demographics, and time-based features, we applied multiple machine learning models to detect and predict fraud. The project focused on effective preprocessing, feature engineering, model evaluation, and visualization to identify fraud risks with high recall.

Requirements
Dataset:
   - https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data
   - Download dataset from the above link.
   - Source: Kaggle – “Financial Transactions Dataset: Analytics”
   - Includes features such as transaction amount, type, card information, demographics, timestamps, and fraud label.
   - Ensure the dataset is available in your working directory before running the notebook.

Prerequisites:

1. Python Environment:
   - Python 3.8 or higher
   - Jupyter Notebook or any IDE that supports `.ipynb` files

2. Required Libraries:
   - pandas
   - numpy
   - seaborn
   - matplotlib
   - scikit-learn
   - xgboost
   - imbalanced-learn (for SMOTE)

Steps to Set Up and Run

1. Open the notebook:
   - Launch `Fraud_Detection.ipynb` in Jupyter or VSCode.

2. Run the cells sequentially:
   - Step 1: Data loading, cleaning, and imputation.
   - Step 2: Feature engineering (e.g., credit utilization, transaction velocity).
   - Step 3: SMOTE oversampling to handle class imbalance.
   - Step 4: Model training (Random Forest, XGBoost, Logistic Regression, Isolation Forest).
   - Step 5: Evaluation using recall, precision, F1-score, and ROC-AUC.
   - Step 6: Visualization and interpretation of fraud patterns.
  
Key Features Implemented

- Data Cleaning: Removal of duplicates, missing value imputation, and normalization of numeric fields.
- Feature Engineering: Custom metrics such as credit utilization, transaction type categorization, outlier detection, and merchant risk scoring.
- Class Balancing: Applied SMOTE to synthesize minority fraud cases.
- Outlier Detection: Used Isolation Forest to flag unusual transactions.
- Time-based Analysis: Evaluated fraud patterns by year, card expiration, and account tenure.

Model Performance Summary

- Random Forest:
  - Validation Recall: 81%, F1: 0.11
  - Test Recall: 2%, Accuracy: 98%
- XGBoost:
  - Validation ROC-AUC: 0.94, Recall: 73%
  - Test ROC-AUC: 0.67, Recall: 3%
- Logistic Regression:
  - Validation Recall: 57%, ROC-AUC: 0.71
- Isolation Forest:
  - ROC-AUC: 0.50 (unsupervised)

Visualizations:

- Feature importance plots
- KDE plots for credit utilization
- Class distribution before/after SMOTE
- Merchant category fraud risk
- Age group vs. fraud trends
- Transaction type vs. fraud occurrence
- Correlation matrix of all features
