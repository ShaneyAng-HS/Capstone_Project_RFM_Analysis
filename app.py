import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime
#from xgboost import XGBRegressor

# 1️⃣ Load the trained KMeans model
kmeans = joblib.load("kmeans_model.pkl")

# 1️⃣ Load the trained KMeans model
rf_model = joblib.load("rf_model.joblib")

# xgb_model = XGBRegressor()
# xgb_model.load_model('xgb_model.json')

# 2️⃣ Page Setup
st.set_page_config(page_title="RFM Cluster & CLV Predictor", layout="centered", page_icon="📊")
st.title("🧠 Customer Segmentation & CLV Prediction")
st.markdown("Input your purchase information to predict the customer cluster and expected monetary value (CLV).")

# 3️⃣ Input Section
st.header("📥 Input Customer Data")
col1, col2 = st.columns(2)
with col1:
    prev_date = st.date_input("Previous Purchase Date", value=datetime(2024, 1, 1))
with col2:
    current_date = st.date_input("Current Purchase Date", value=datetime(2025, 1, 1))

recency = (current_date - prev_date).days
st.markdown(f"🕒 **Calculated Recency:** `{recency}` days")

frequency = st.number_input("Frequency (Number of Orders)", min_value=1, value=5)
monetary = st.number_input("Actual Monetary Spend ($)", min_value=1.0, value=500.0)
total_qty = st.number_input("Total Quantity Purchased", min_value=1, value=25)
tenure = st.number_input("Customer Tenure (days)", min_value=1, value=365)

# Derived features for regression
#avg_order_value = monetary / frequency
avg_qty_per_order = total_qty / frequency

# Log-transformed features for clustering
log_recency = np.log1p(recency)
log_frequency = np.log1p(frequency)
log_monetary = np.log1p(monetary)

# Buttons
if st.button("🚀 Predict Cluster & CLV"):
    # --- Predict Cluster --- #
    user_rfm = np.array([[log_recency, log_frequency, log_monetary]])
    cluster = kmeans.predict(user_rfm)[0]

    # --- Predict Monetary --- #
    input_df = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        #'AvgOrderValue': [avg_order_value],
        'TotalQuantity': [total_qty],
        'AvgQuantityPerOrder': [avg_qty_per_order],
        'Tenure': [tenure]
    })
    predicted_monetary = rf_model.predict(input_df)[0]

    # --- Output Results --- #
    st.subheader("📊 Prediction Results")
    st.success(f"✅ This customer belongs to **Cluster {cluster}**.")

    cluster_notes = {
        0: "🟣 Possibly loyal or high-value customers.",
        1: "🔵 Maybe new or potential customers.",
        2: "🟠 Inactive or low-priority customers.",
        3: "🟢 Core or top-tier customers."
    }
    st.markdown(cluster_notes.get(cluster, "ℹ️ No specific description for this cluster."))

    #st.info(f"💰 **Actual Spend:** ${monetary:.2f}")
    st.success(f"📈 **Predicted Spend (CLV):** ${predicted_monetary:.2f}")


