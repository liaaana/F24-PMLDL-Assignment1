import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI
st.title("Wine Quality Classifier")

# Input fields 
alcohol = st.number_input("Alcohol", min_value=0.0)
malic_acid = st.number_input("Malic Acid", min_value=0.0)
ash = st.number_input("Ash", min_value=0.0)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", min_value=0.0)
magnesium = st.number_input("Magnesium", min_value=0)
total_phenols = st.number_input("Total Phenols", min_value=0.0)
flavanoids = st.number_input("Flavanoids", min_value=0.0)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", min_value=0.0)
proanthocyanins = st.number_input("Proanthocyanins", min_value=0.0)
color_intensity = st.number_input("Color Intensity", min_value=0.0)
hue = st.number_input("Hue", min_value=0.0)
od280_od315_of_diluted_wines = st.number_input("OD280/OD315 of Diluted Wines", min_value=0.0)
proline = st.number_input("Proline", min_value=0)

# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        "alcohol": alcohol,
        "malic_acid": malic_acid,
        "ash": ash,
        "alcalinity_of_ash": alcalinity_of_ash,
        "magnesium": magnesium,
        "total_phenols": total_phenols,
        "flavanoids": flavanoids,
        "nonflavanoid_phenols": nonflavanoid_phenols,
        "proanthocyanins": proanthocyanins,
        "color_intensity": color_intensity,
        "hue": hue,
        "od280_od315_of_diluted_wines": od280_od315_of_diluted_wines,
        "proline": proline
    }
    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"The model predicts class: {prediction}")
    else:
        st.error(f"The model failed to predict class")
