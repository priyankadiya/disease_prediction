import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model and other necessary files
model = joblib.load('Health_Disease_RandomForestModel.joblib')
df1 = pd.read_csv('Health_dataset/Symptom-severity.csv')
discrp = pd.read_csv('Health_dataset/symptom_Description.csv')
prec = pd.read_csv('Health_dataset/symptom_precaution.csv')

# Title for Streamlit App
st.title("Health Disease Prediction System")

# Create user inputs for 17 symptoms using text inputs in Streamlit
symptoms = []
for i in range(1, 18):
    symptom = st.text_input(f"Enter Symptom {i}", "").strip().lower()
    if symptom == "":  # If symptom is left empty, set it to None (null)
        symptom = None
    symptoms.append(symptom)

# Create a dictionary for mapping symptoms to weights
symptom_weights = dict(zip(df1['Symptom'].str.lower().str.strip(), df1['weight']))

# Function to process symptoms and make predictions
def predict_disease(symptoms):
    processed_symptoms = []
    for symptom in symptoms:
        if symptom and symptom in symptom_weights:  # Only process non-null symptoms
            processed_symptoms.append(symptom_weights[symptom])
        else:
            processed_symptoms.append(0)  # For missing symptoms, append 0

    # Convert the processed symptoms into a 2D numpy array for prediction
    processed_symptoms = np.array(processed_symptoms).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(processed_symptoms)[0]

    # Get the disease description and precautions
    description = discrp[discrp['Disease'] == prediction]['Description'].values[0]
    precautions = prec[prec['Disease'] == prediction].iloc[0, 1:].dropna().tolist()

    return prediction, description, precautions

# Button for prediction
if st.button("Predict"):
    prediction, description, precautions = predict_disease(symptoms)
    
    # Display the results
    st.write(f"**Predicted Disease**: {prediction}")
    st.write(f"**Description**: {description}")
    st.write(f"**Precautions**: {', '.join(precautions)}")
