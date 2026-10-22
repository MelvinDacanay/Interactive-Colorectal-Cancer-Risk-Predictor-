
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "age": 45,
        "gender": "Male",
        "bmi": 25.0,
        "lifestyle": "Moderate Exercise",
        "ethnicity": "Asian",
        "family_history_crc": "No",
        "pre_existing_conditions": "N/A",
        "carbohydrates_g": 250,
        "proteins_g": 75,
        "fats_g": 65,
        "vitamin_a_iu": 3000,
        "vitamin_c_mg": 90,
        "iron_mg": 8,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_class"] in [0, 1]
    assert result["experimental_label"] in ["At Risk", "No Risk"]
    assert "disclaimer" in result


def test_invalid_age_is_rejected():
    response = client.post(
        "/predict",
        json={
            "age": 150,
            "gender": "Male",
            "bmi": 25.0,
            "lifestyle": "Moderate Exercise",
            "ethnicity": "Asian",
            "family_history_crc": "No",
            "pre_existing_conditions": "N/A",
            "carbohydrates_g": 250,
            "proteins_g": 75,
            "fats_g": 65,
            "vitamin_a_iu": 3000,
            "vitamin_c_mg": 90,
            "iron_mg": 8,
        },
    )

    assert response.status_code == 422
