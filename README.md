# Medical Charges Prediction

A machine learning pipeline for predicting medical insurance charges using multiple regression models with cross-validation, hyperparameter tuning, and automated model selection.

## Project Overview

This project implements an end-to-end ML pipeline that:
- Ingests and preprocesses the Kaggle Medical Insurance dataset
- Performs exploratory data analysis with KDE visualizations
- Conducts statistical significance testing (Pearson correlation, t-tests)
- Trains four regression models: Linear Regression, Elastic Net, Random Forest, and XGBoost
- Uses 5-fold stratified cross-validation with grid search for hyperparameter optimization
- Automatically selects the champion model based on lowest AMSE (Average Mean Squared Error)
- Evaluates the champion on a held-out test set
- Saves trained models as pickles and generates HTML reports

## Results Summary

### Champion Model: XGBoost Regressor

| Metric | Value |
|--------|-------|
| **CV AMSE** | 20,627,344.29 |
| **Test MSE** | 18,178,220.38 |
| **Test RMSE** | 4,263.59 |
| **Test MAE** | 2,428.38 |
| **Test R²** | 0.8829 |
| **Test Adjusted R²** | 0.8793 |

### Hyperparameters
```
colsample_bytree: 1.0
learning_rate: 0.03
max_depth: 3
n_estimators: 200
subsample: 0.8
```

### Model Comparison (CV AMSE)
1. **XGBoost** – 20,627,344.29 ✓ *Champion*
2. Random Forest – 21,878,048.78
3. Elastic Net – 38,114,999.64
4. Linear Regression – 38,118,421.55

## Project Structure

```
Medical-Charges-Prediction/
├── src/
│   ├── main.py                           # Entry point: runs full pipeline
│   ├── utils/
│   │   ├── config.py                     # Logger setup & project paths
│   │   └── helpers.py                    # Shared CV, GS, metrics utilities
│   ├── ingestion_preprocessing/
│   │   ├── extract_dataset.py            # Download Kaggle dataset
│   │   └── split_encode.py               # Split, encode, normalize & save train/test
│   ├── eda/
│   │   ├── feature_analysis.py           # KDE distribution plots (HTML)
│   │   └── stat_tests.py                 # Pearson & t-test significance
│   ├── modeling/
│   │   ├── lr.py                         # Linear Regression
│   │   ├── en.py                         # Elastic Net
│   │   ├── rfr.py                        # Random Forest Regressor
│   │   ├── xgbr.py                       # XGBoost Regressor
│   │   └── champ.py                      # Champion selection & test evaluation
│   └── evaluation/
│       ├── model_train_results/          # Per-model metric text files
│       └── champ_model_test/             # Champion test HTML report
├── database/
│   ├── raw/                              # Raw Kaggle dataset
│   ├── train_encoded.csv                 # Preprocessed train set
│   └── test_encoded.csv                  # Preprocessed test set
├── artifacts/
│   ├── models/                           # Trained model pickles
│   │   ├── lr.pkl
│   │   ├── en.pkl
│   │   ├── rfr.pkl
│   │   ├── xgbr.pkl
│   │   └── champ.pkl
│   └── cv_results/                       # CV performance HTML charts
├── figures/                              # EDA plots (HTML)
├── logs/                                 # Timestamped execution logs
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd Medical-Charges-Prediction
```

2. Create a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python -m src.main
```

This will:
1. Download & preprocess the dataset (split 80/20, encode categorical features, normalize)
2. Train 4 models with 5-fold stratified CV + grid search
3. Select the champion model (XGBoost)
4. Evaluate on test set
5. Save model pickles & HTML reports

### Individual Steps

Extract dataset:
```bash
python -m src.ingestion_preprocessing.extract_dataset
```

Preprocess & split:
```bash
python -m src.ingestion_preprocessing.split_encode
```

Train individual models:
```bash
python -m src.modeling.lr
python -m src.modeling.en
python -m src.modeling.rfr
python -m src.modeling.xgbr
```

Select champion & evaluate:
```bash
python -m src.modeling.champ
```

## Outputs

- **Model Pickles** – `artifacts/models/*.pkl` (trained estimators for inference)
- **CV Performance Charts** – `artifacts/cv_results/*.html` (MSE, RMSE, MAE, R², Adjusted R² by fold)
- **EDA Plots** – `figures/*.html` (feature distributions via KDE)
- **Champion Report** – `src/evaluation/champ_model_test/champ_model_test_scores.html`
- **Training Metrics** – `src/evaluation/model_train_results/*.txt` (per-model test scores)
- **Logs** – `logs/YYYY-MM-DD/*.log` (timestamped execution traces)

## Dependencies

- **pandas** – Data manipulation
- **numpy** – Numerical computations
- **scipy** – Statistical tests
- **scikit-learn** – ML models, preprocessing, CV, metrics
- **xgboost** – XGBoost regressor
- **plotly** – Interactive HTML visualizations
- **kagglehub** – Kaggle dataset API

## Notes

- Dataset is split 80/20 (train/test) before any modeling
- Region categorical feature is one-hot encoded
- Numeric features are min-max normalized
- Stratified 5-fold CV uses binned target (qcut) for regression stratification
- AMSE selection criterion: lower is better (average MSE across CV folds)
- All models saved as sklearn-compatible pickles for reproducible inference

