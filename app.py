import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained models
kmeans = joblib.load("kmeans_model.pkl")
rf_model = joblib.load("rf_model.joblib")

# Page Setup
st.set_page_config(page_title="Customer Insights Portal", layout="centered", page_icon="📊")

# Custom CSS to change background color and style
custom_css = """
<style>
    .stApp {
        background-color: #000080;
    }
    section[data-testid="stSidebar"] {
        background-color: #000080;
    }
    h1, h2, h3, h4 {
        color: #333333;
    }
    div[data-testid="metric-container"] {
        background-color: #000080;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("📊 Customer Segmentation & Lifetime Value Estimator")
st.markdown("This tool classifies your customers into strategic segments and estimates their Customer Lifetime Value (CLV) based on transaction patterns.")

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
monetary = st.number_input("Total Monetary Spend ($)", min_value=1.0, value=500.0, help="Total spend to date")
total_qty = st.number_input("Total Quantity Purchased", min_value=1, value=25)
tenure = st.number_input("Customer Tenure (days)", min_value=1, value=365, help="How long the customer has been with the company")

# Feature Engineering
avg_qty_per_order = total_qty / frequency
log_recency = np.log1p(recency)
log_frequency = np.log1p(frequency)
log_monetary = np.log1p(monetary)

# Prediction Trigger
if st.button("📈 Generate Insights"):
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
    "<div style='text-align: center; color: grey;'>"
    "Developed by <b>Shaney Ang Tech</b> | Powered by Streamlit | © 2025"
    "</div>",
    unsafe_allow_html=True
)
