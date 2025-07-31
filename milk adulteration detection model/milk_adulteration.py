import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

st.set_page_config(
    page_title="Milk Adulteration Detection Model",
    page_icon="🥛",
    layout="wide"
)


# --- Title ---
#st.title("Milk Adulteration Prediction")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Milk Adulteration Prediction")

st.image('milk.jpeg', use_container_width=True)

st.write("""
Please note that this model has  been specifically trained to detect the presence of five different types of adulterants in milk - bicarbonates, starch, sucrose, formaldehyde 
and peroxide. It relies on all 518 spectroscopic measurements across different wavelengths (SPC 1002 - SPC 2997) to give the most accurate 
prediction.
""")

# --- Load Model and PCA ---
@st.cache_resource
def load_model_scaler_and_pca():
    with open("log_reg_m.pkl", "rb") as f_model:
        model = pickle.load(f_model)
    with open("scaler_m.pkl", "rb") as f_scaler:
        scaler = pickle.load(f_scaler)
    with open("pca_m.pkl", "rb") as f_pca:
        pca = pickle.load(f_pca)
        
    return model, scaler, pca

model, scaler, pca = load_model_scaler_and_pca()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV with spectral features (e.g., SPC1002–SPC2997)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # --- Select spectral columns only ---
    spectral_cols = [col for col in df.columns if col.startswith("SPC")]

    if len(spectral_cols) == 0:
        st.error("No spectral (SPCxxxx) columns found in uploaded data.")
    elif len(spectral_cols) < 518:
        st.error("Some spectral (SPCxxxx) columns are missng in uploaded data. Please upload all 518 spectroscopic results")
    else:
        # --- PCA Transform ---
        try:
            X_raw = df[spectral_cols]
            X_scaled=scaler.transform(X_raw)
            X_pca = pca.transform(X_scaled)

            # --- Model Prediction ---
            predictions = model.predict(X_pca)
            probabilities = model.predict_proba(X_pca)

            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "Raw Milk", 1: "Adulterated Milk"})
            #df["Confidence"] = np.max(probabilities, axis=1)
            df["Confidence"] = (np.max(probabilities, axis=1) * 100).round(2).astype(str) + "%"
            #st.write(df[["Result", "Confidence"]])

            st.subheader("Prediction Results")
            st.dataframe(df[["Prediction", "Confidence"]])

            # --- Download Results ---
            st.download_button("Download Predictions", df.to_csv(index=False), file_name="milk_predictions.csv")

        except Exception as e:
            st.error(f"Error during PCA or prediction: {e}")

