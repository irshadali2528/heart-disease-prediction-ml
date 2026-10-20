
# Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to classify heart disease status based on health-related features from the UCI Heart Disease dataset.

A Logistic Regression model is trained and evaluated using Python and Scikit-learn.

## Objectives

- Load and explore the Heart Disease dataset.
- Preprocess data and handle missing values.
- Train a Logistic Regression classification model.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Generate predictions on unseen test data.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Heart Disease (ID 45)
- **Records:** 303
- **Input features:** 13
- **Target:** Binary classification

For this project, target values are grouped into two classes:
- `0`: No disease indicated in the dataset
- `1`: Disease indicated in the dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- UCI ML Repository Python package

## Model

**Algorithm:** Logistic Regression

The workflow includes:
- Train-test split (80% training, 20% testing)
- Median imputation for missing values
- Feature standardization
- Model training and evaluation

## Results

- **Test Accuracy:** 86.89%
- **Test Records:** 61
- **Disease-class Precision:** 81%
- **Disease-class Recall:** 93%
- **Disease-class F1-score:** 87%

These are results from one train-test split. Performance may vary with different splits or datasets.

## Project Files

- `heart_disease_prediction.py` — Source code for training and evaluation
- `heart_disease_model.pkl` — Trained model pipeline
- `missing_value_imputer.pkl` — Saved missing-value imputer

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python heart_disease_prediction.py
```

The script downloads the dataset from the UCI repository, trains the model, evaluates it, and saves the model files.

## Disclaimer

This is an educational machine learning project. It is not intended for clinical use, medical diagnosis, or treatment decisions.

## Author

Irshad Ali
**Machine Learning | Python | Scikit-learn**
