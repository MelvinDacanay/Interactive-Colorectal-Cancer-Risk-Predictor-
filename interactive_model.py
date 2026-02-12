import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ohe = OneHotEncoder()
scaler = StandardScaler()
estimator = RandomForestClassifier()

transformer = ColumnTransformer(transformers=[
    ('ohe', ohe, ['Gender', 'Lifestyle', 'Ethnicity', 'Family_History_CRC', 'Pre-existing Conditions']),
    ('scaler', scaler, ['Age', 'BMI', 'Carbohydrates (g)', 'Proteins (g)', 'Fats (g)', 'Vitamin A (IU)', 'Vitamin C (mg)', 'Iron (mg)'])
])

# pipeline init and model fitting

pl = make_pipeline(transformer, estimator)

df = pd.read_csv('clean.csv')
X = df.iloc[:,0:13]
y = df.iloc[:,13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
pl.fit(X_train, y_train)

# Data Input

age = input('Type your age: ')
gender = input('Type your gender (Male/Female): ')
BMI = input('Type your BMI: ')
Lifestyle = input('Lifestyle (Sedentary, Smoker, Moderate Exercise, Active): ')
Ethnicity = input('Ethnicity (Hispanic, Asian, African American, Caucasian): ')
Family_History_CRC = input('Family_History_CRC (Yes/No): ')
Pre_existing_Conditions = input('Pre-existing Conditions (Diabetes, Obesity, N/A, Hypertension): ')
Iron_intake = input('Iron Daily Intake (mg): ')
Protein_intake = input('Proteins Daily Intake (g): ')
Fats_intake = input('Fats Daily Intake (g): ')
VitaminA_intake = input('Vitamin A Daily Intake (IU):')
VitaminC_Intake = input('vitaminC Daily Intake (mg): ')
Carb_intake = input('carbs Daily Intake (g): ')

# Initialization of test observation to form prediction

new_person = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    'BMI': BMI,
    'Lifestyle': Lifestyle,
    "Ethnicity": Ethnicity,
    'Family_History_CRC': Family_History_CRC,
    'Pre-existing Conditions': Pre_existing_Conditions,
    'Iron (mg)': Iron_intake,
    'Proteins (g)': Protein_intake,
    'Fats (g)': Fats_intake,
    'Vitamin A (IU)': VitaminA_intake,
    'Vitamin C (mg)': VitaminC_Intake,
    'Carbohydrates (g)': Carb_intake
}])
y_pred =  pl.predict(new_person)
if y_pred == 0:
    print('No Risk')
else:
    print('At Risk')