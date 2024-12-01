import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('insurance_model.pkl')

# Streamlit app title
st.title("Insurance Cost Prediction App")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Convert inputs to the model's format
sex_male = 1 if sex == "male" else 0
smoker_yes = 1 if smoker == "yes" else 0
region_encoded = {
    "northeast": [0, 0, 0],
    "northwest": [1, 0, 0],
    "southeast": [0, 1, 0],
    "southwest": [0, 0, 1],
}[region]

# Combine all inputs
inputs = np.array([[age, bmi, children, sex_male, smoker_yes] + region_encoded])

# Predict insurance cost
if st.button("Predict"):
    prediction = model.predict(inputs)[0]
    st.success(f"The predicted insurance cost is: ${prediction:.2f}")
