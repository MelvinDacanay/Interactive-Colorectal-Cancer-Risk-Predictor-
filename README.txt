# Interactive Colorectal Cancer Risk Predictor

## Overview

An experimental machine-learning project that exposes a colorectal cancer risk prediction model through a REST API. The project demonstrates model training, preprocessing pipelines, API development, automated testing, and Docker containerization.

**Disclaimer:** This is an educational project, not a medical diagnostic tool. It must not be used to diagnose, rule out, or make clinical decisions about colorectal cancer, and it does not replace recommended screening or professional medical advice.

## Tech Stack

- Python 3.13
- pandas
- scikit-learn
- FastAPI
- Uvicorn
- pytest
- Docker

## Project Structure

- `train_model.py` — trains and evaluates the model.
- `crc_model.joblib` — saved preprocessing and model pipeline.
- `app.py` — REST API with health-check and prediction endpoints.
- `test_app.py` — automated API tests.
- `requirements.txt` — pinned Python dependencies.
- `Dockerfile` — builds the API container.
- `.dockerignore` — excludes unnecessary files from the Docker build.
- `data/clean.csv` — dataset used for training.

## Model Evaluation

The model was evaluated on a held-out test set of 200 records.

- Accuracy: 91%
- Positive-class recall: 42%
- Positive-class precision: 100%

The dataset is imbalanced, and the model missed a substantial proportion of positive examples in this test split. Accuracy alone is therefore insufficient to establish model quality. These results do not demonstrate clinical validity or suitability for medical use.

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app:app --reload
```

Open `http://localhost:8000/docs` for interactive API documentation.

## Run Tests

```bash
python -m pytest -v
```

## Run with Docker

Build the image:

```bash
docker build -t crc-risk-api .
```

Start the container:

```bash
docker run --name crc-risk-container -p 8000:8000 crc-risk-api
```

Visit `http://localhost:8000/health` to check the API, or `http://localhost:8000/docs` to explore its endpoints.

To stop the container, press `Ctrl+C` in the terminal running it. To remove the stopped container:

```bash
docker rm crc-risk-container
```

## API Endpoints

- `GET /health` — reports API health and model-loading status.
- `POST /predict` — accepts input features and returns an experimental model prediction.

## Future Improvements

- Improve positive-class recall and evaluate false-negative rates.
- Add input validation for allowed categorical values.
- Automate testing and builds with a CI/CD workflow.
- Deploy the container to AWS.
- Add monitoring, logging, and model-version tracking.





1. Question I had coming into this
- Based off of basic input information could a model 
predict if an individual is at risk for CRC Based
off their diet

2. Why does it matter
- A person could screen themselves and see if they
are categorized as someone who may be at risk.
This could be helpful for someone who wants to
be made aware of if their diet is problematic

3. data
- The data was downloaded from kaggle
- The data has hundreds of observations 
- The key features of the data include their
history, and details about their diet.

4. method
- I chose a random forest classifier because it 
excels at classifying data. It can get very powerful
as it uses several trees instead of just one.
- I evaluated it via confusion matrix and accuracy score
- The former to guage it quantitatively and the latter
for quick evaluation of how the model did
- I also cross validated this model to compare it
to other folds and prevent cherry picking and just ensure
I was getting the best representation of both may
model and my data
- I used a column transformer combined with
a pipeline to reduce data leakage and ease
the training process

5. Results
- The model scored .88 on its first test so
i was already able to implement it.
- There were several false negetives and this
may have to do with the fact that there are more
variables that have to do with a patients risk
level
    - This also may be the reason why a 3d PCA
    analysis appeared to only have one distinct
    cluster

6. Takeaways 
- I discovered that BMI was the biggest mover in
whether a person was at risk or not
- Vitamin intakes did not have much effect on A
persons risk level
