
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# 1. Define paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "clean.csv"
MODEL_PATH = BASE_DIR / "crc_model.joblib"


# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CRC_Risk"])
y = df["CRC_Risk"]


# 3. Define the preprocessing steps
categorical_features = [
    "Gender",
    "Lifestyle",
    "Ethnicity",
    "Family_History_CRC",
    "Pre-existing Conditions",
]

numeric_features = [
    "Age",
    "BMI",
    "Carbohydrates (g)",
    "Proteins (g)",
    "Fats (g)",
    "Vitamin A (IU)",
    "Vitamin C (mg)",
    "Iron (mg)",
]

transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("scaler", StandardScaler(), numeric_features),
    ]
)


# 4. Create the complete ML pipeline
pipeline = make_pipeline(
    transformer,
    RandomForestClassifier(random_state=42)
)


# 5. Split data and train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline.fit(X_train, y_train)


# 6. Evaluate the model on held-out data
predictions = pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))


# 7. Save the complete pipeline
joblib.dump(pipeline, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
