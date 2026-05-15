import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

st.title("Sensor Fault Detection System")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df[[col for col in df.columns if 'sensor' in col]]

    scaler = joblib.load("models/scaler.pkl")
    X = scaler.transform(X)
    
    predictions = model.predict(X)
    df['Prediction'] = predictions
    
    st.write(df)
    
    if 1 in predictions:
        st.error("Fault Detected!")
    else:
        st.success("All Sensors Normal")

