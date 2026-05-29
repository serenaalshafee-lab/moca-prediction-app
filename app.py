
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("moca_random_forest_model.pkl")

st.title("MoCA Score Prediction Tool")

st.write(
    "This tool predicts the MoCA score for stroke patients based on selected "
    "demographic, clinical, sleep, and psychological predictors."
)

age = st.number_input("Age", min_value=18, max_value=100, value=60)

gender_label = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)

education_label = st.selectbox(
    "Educational Level",
    options=["High School", "Diploma", "Bachelor", "Postgraduate"]
)

heart_disease_label = st.selectbox(
    "Heart Disease",
    options=["No", "Yes"]
)

sleep_score = st.number_input(
    "PSQI Total Score",
    min_value=0.0,
    max_value=30.0,
    value=9.0
)

anxiety_score = st.number_input(
    "Anxiety Total Score",
    min_value=0.0,
    max_value=42.0,
    value=18.0
)

# Convert labels back to dataset codes
gender_code = 1 if gender_label == "Male" else 2

education_code = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor": 3,
    "Postgraduate": 4
}[education_label]

heart_disease_code = 1 if heart_disease_label == "Yes" else 0

if st.button("Predict MoCA Score"):
    new_patient = pd.DataFrame({
        "age": [age],
        "gender": [gender_code],
        "education": [education_code],
        "Heart_disease": [heart_disease_code],
        "Sleep_scores": [sleep_score],
        "Anxiety_score": [anxiety_score]
    })

    prediction = model.predict(new_patient)[0]

    st.success(f"Predicted MoCA score: {prediction:.2f}")

    st.caption(
        "Note: This prediction is intended to support clinical decision-making "
        "and does not replace formal cognitive assessment."
    )
