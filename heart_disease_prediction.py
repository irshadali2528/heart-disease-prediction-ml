
import pandas as pd
import joblib
import os

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.iloc[:, 0].copy()

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

valid_rows = y.notna()
X = X.loc[valid_rows]
y = (y.loc[valid_rows] > 0).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test), columns=X.columns
)

# Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Test Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and preprocessing tool
os.makedirs("heart_disease_project", exist_ok=True)

joblib.dump(model, "heart_disease_project/heart_disease_model.pkl")
joblib.dump(imputer, "heart_disease_project/missing_value_imputer.pkl")

print("\nPython code executed successfully!")
