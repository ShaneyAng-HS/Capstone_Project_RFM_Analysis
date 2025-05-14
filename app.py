import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained models
kmeans = joblib.load("kmeans_model.pkl")
rf_model = joblib.load("rf_model.joblib")

# Set page config
st.set_page_config(page_title="Customer Insights Portal", layout="centered", page_icon="🧠")

# Apply enhanced dark theme with better contrast
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: #eaeaea;
        }
        h1, h2, h3, h4, h5 {
            color: #ffffff;
        }
        .stTextInput > div > input,
        .stNumberInput input,
        .stDateInput input {
            background-color: #444444;
            color: #ffffff;
            border-radius: 6px;
        }
        .stButton > button {
            background-color: #4db8ff;
            color: black;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #6dccff;
            color: black;
        }
        .stMarkdown {
            color: #eaeaea;
        }
        .metric {
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and intro
st.title("🧠 Customer Segmentation & CLV Estimator")
st.markdown("This tool classifies customers into strategic segments and predicts Customer Lifetime Value (CLV) based on purchase behavior.")

# Input section
st.header("🛒 Customer Purchase Details")
col1, col2 = st.columns(2)

with col1:
    prev_date = st.date_input("Previous Purchase Date", value=datetime(2024, 1, 1))
with col2:
    current_date = st.date_input("Most Recent Purchase Date", value=datetime(2025, 1, 1))

recency = (current_date - prev_date).days
st.markdown(f"📆 **Recency (days since last purchase):** `{recency}`")

frequency = st.number_input("🧾 Purchase Frequency", min_value=1, value=5)
monetary = st.number_input("💵 Total Spend ($)", min_value=1.0, value=500.0)
total_qty = st.number_input("📦 Total Quantity Purchased", min_value=1, value=25)
tenure = st.number_input("⏳ Customer Tenure (days)", min_value=1, value=365)

# Feature Engineering
avg_qty_per_order = total_qty / frequency
log_recency = np.log1p(recency)
log_frequency = np.log1p(frequency)
log_monetary = np.log1p(monetary)

# Prediction Trigger
if st.button("🚀 Generate Insights"):
    user_rfm = np.array([[log_recency, log_frequency, log_monetary]])
    cluster = kmeans.predict(user_rfm)[0]

    input_df = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'TotalQuantity': [total_qty],
        'AvgQuantityPerOrder': [avg_qty_per_order],
        'Tenure': [tenure]
    })
    predicted_monetary = rf_model.predict(input_df)[0]

    # Result section
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
    "<div style='text-align: center; color: #888;'>"
    "Developed by <b>Shaney Ang Tech</b> | Powered by Streamlit | © 2025"
    "</div>",
    unsafe_allow_html=True
)
