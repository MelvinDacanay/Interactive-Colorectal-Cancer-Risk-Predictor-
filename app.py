
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate and load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crc_model.joblib"

pipeline = joblib.load(MODEL_PATH)

# Create the API application
app = FastAPI(
    title="CRC Risk Predictor API",
    description=(
        "An experimental machine-learning demonstration. "
        "Not for clinical use or diagnosis."
    ),
    version="1.0.0",
)


# Define the data a prediction request must contain
class PredictionInput(BaseModel):
    age: float = Field(ge=0, le=120)
    gender: str
    bmi: float = Field(gt=0, le=100)
    lifestyle: str
    ethnicity: str
    family_history_crc: str
    pre_existing_conditions: str
    carbohydrates_g: float = Field(ge=0)
    proteins_g: float = Field(ge=0)
    fats_g: float = Field(ge=0)
    vitamin_a_iu: float = Field(ge=0)
    vitamin_c_mg: float = Field(ge=0)
    iron_mg: float = Field(ge=0)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: PredictionInput):
    # Convert API input names into the exact training column names
    person = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "BMI": data.bmi,
        "Lifestyle": data.lifestyle,
        "Ethnicity": data.ethnicity,
        "Family_History_CRC": data.family_history_crc,
        "Pre-existing Conditions": data.pre_existing_conditions,
        "Carbohydrates (g)": data.carbohydrates_g,
        "Proteins (g)": data.proteins_g,
        "Fats (g)": data.fats_g,
        "Vitamin A (IU)": data.vitamin_a_iu,
        "Vitamin C (mg)": data.vitamin_c_mg,
        "Iron (mg)": data.iron_mg,
    }])

    prediction = int(pipeline.predict(person)[0])

    return {
        "predicted_class": prediction,
        "experimental_label": (
            "At Risk" if prediction == 1 else "No Risk"
        ),
        "disclaimer": (
            "Experimental model output, not a diagnosis. "
            "This result cannot rule out colorectal cancer "
            "or replace recommended screening."
        ),
    }
