import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained models
kmeans = joblib.load("kmeans_model.pkl")
rf_model = joblib.load("rf_model.joblib")

# App Config with professional color theme
st.set_page_config(page_title="Customer Insights Portal", layout="centered", page_icon="🧠")

# Apply custom dark theme using Markdown (no white background)
st.markdown(
    """
    <style>
        body {
            background-color: #2b2b2b;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #2b2b2b;
            color: #f0f0f0;
        }
        .stButton > button {
            background-color: #4db8ff;
            color: black;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton > button:hover {
            background-color: #73ccff;
            color: black;
        }
        .stNumberInput, .stDateInput {
            background-color: #3b3b3b !important;
            color: #f0f0f0 !important;
        }
        .stTextInput > div > input {
            color: #f0f0f0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and intro
st.title("🧠 Customer Segmentation & CLV Estimator")
st.markdown("Use this tool to classify your customers into strategic segments and estimate their lifetime value based on purchase behavior.")

# Input Section
st.header("🛒 Customer Purchase Details")
col1, col2 = st.columns(2)

with col1:
    prev_date = st.date_input("Previous Purchase Date", value=datetime(2024, 1, 1))
with col2:
    current_date = st.date_input("Most Recent Purchase Date", value=datetime(2025, 1, 1))

recency = (current_date - prev_date).days
st.markdown(f"📆 **Recency (days since last purchase):** `{recency}`")

frequency = st.number_input("Purchase Frequency", min_value=1, value=5, help="Number of orders placed to date")
monetary = st.number_input("Total Monetary Spend ($)", min_value=1.0, value=500.0, help="Total spend to date in USD")
total_qty = st.number_input("Total Quantity Purchased", min_value=1, value=25)
tenure = st.number_input("Customer Tenure (days)", min_value=1, value=365, help="How long the customer has been with the company")

# Feature Engineering
avg_qty_per_order = total_qty / frequency
log_recency = np.log1p(recency)
log_frequency = np.log1p(frequency)
log_monetary = np.log1p(monetary)

# Prediction Trigger
if st.button("🚀 Generate Insights"):
    # Predict cluster
    user_rfm = np.array([[log_recency, log_frequency, log_monetary]])
    cluster = kmeans.predict(user_rfm)[0]

    # Predict monetary value
    input_df = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'TotalQuantity': [total_qty],
        'AvgQuantityPerOrder': [avg_qty_per_order],
        'Tenure': [tenure]
    })
    predicted_monetary = rf_model.predict(input_df)[0]

    # Display Results
    st.subheader("🎯 Predicted Segment & Value")

    segment_map = {
        0: "🟣 New/Moderate Buyers",
        1: "🔴 At-Risk",
        2: "🟠 High Potential",
        3: "🟢 VIPs"
    }

    descriptions = {
        0: "These customers show moderate engagement. Nurture with consistent marketing.",
        1: "These customers may be slipping away. Consider re-engagement campaigns.",
        2: "These are promising leads. Focused promotion may convert them to loyal buyers.",
        3: "Top-tier loyal customers. Prioritize retention and premium experiences."
    }

    st.success(f"🧩 **Customer Segment:** {segment_map.get(cluster)}")
    st.info(descriptions.get(cluster))

    st.metric("💰 Predicted Lifetime Value (CLV)", f"${predicted_monetary:,.2f}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #a9a9a9;'>"
    "Developed by <b>Shaney Ang Tech</b> | Powered by Streamlit | © 2025"
    "</div>",
    unsafe_allow_html=True
)
