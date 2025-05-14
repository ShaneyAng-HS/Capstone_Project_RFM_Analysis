# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime

# # Load trained models
# kmeans = joblib.load("kmeans_model.pkl")
# rf_model = joblib.load("rf_model.joblib")

# # Page Setup
# st.set_page_config(page_title="Customer Insights Portal", layout="centered", page_icon="📊")

# # Custom CSS to change background color and style
# custom_css = """
# <style>
#     .stApp {
#         background-color: #e6f0ff;
#         color: black;
#     }

#     section[data-testid="stSidebar"] {
#         background-color: #e6f0ff;
#         color: black;
#     }

#     h1, h2, h3, h4, h5, h6, p, div, span, label {
#         color: black !important;
#     }

#     div[data-testid="metric-container"] {
#         background-color: #ffffff;
#         color: black;
#         padding: 10px;
#         border-radius: 10px;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }

#     input, textarea, select, div[data-baseweb="input"] input {
#         background-color: #ffffff !important;
#         color: black !important;
#         border: 1px solid #ccc;
#         border-radius: 5px;
#     }

#     div[data-baseweb="input"] button {
#         background-color: #cccccc !important;
#         color: black !important;
#         border: 1px solid #999 !important;
#         border-radius: 0px 5px 5px 0px !important;
#     }

#     button, div[data-baseweb="button"] {
#         background-color: #cccccc !important;
#         color: black !important;
#         border: 1px solid #999 !important;
#         border-radius: 5px !important;
#     }

#     div[data-baseweb="select"] {
#         background-color: #ffffff !important;
#         color: black !important;
#     }

#     div[class*="stDateInput"] input {
#         background-color: #ffffff !important;
#         color: black !important;
#     }

#     /* Calendar popover */
#     div[data-baseweb="datepicker-popover"] {
#         background-color: #cccccc !important;
#         color: black !important;
#     }

#     /* Fix navigation buttons inside calendar popover */
#     div[data-baseweb="datepicker-popover"] div[data-baseweb="calendar-header"] button {
#         background-color: #bbbbbb !important;
#         color: black !important;
#         border: 1px solid #999 !important;
#         border-radius: 5px !important;
#     }

#     div[data-baseweb="calendar"] {
#         background-color: #cccccc !important;
#         color: black !important;
#     }

#     div[data-baseweb="calendar"] div[role="option"][aria-selected="true"] {
#         background-color: #999999 !important;
#         color: white !important;
#         border-radius: 50% !important;
#     }

#     div[data-baseweb="calendar-header"] {
#         background-color: #cccccc !important;
#         color: black !important;
#     }
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# st.title("📊 Customer Segmentation & Lifetime Value Estimator")
# st.markdown("This tool classifies your customers into strategic segments and estimates their Customer Lifetime Value (CLV) based on transaction patterns.")

# # Input Section
# st.header("🛒 Customer Purchase Details")
# col1, col2 = st.columns(2)

# with col1:
#     prev_date = st.date_input("Previous Purchase Date", value=datetime(2024, 1, 1))
# with col2:
#     current_date = st.date_input("Most Recent Purchase Date", value=datetime(2025, 1, 1))

# recency = (current_date - prev_date).days
# st.markdown(f"📆 **Recency (days since last purchase):** `{recency}`")

# frequency = st.number_input("Purchase Frequency", min_value=1, value=5, help="Number of orders placed to date")
# monetary = st.number_input("Total Monetary Spend ($)", min_value=1.0, value=500.0, help="Total spend to date")
# total_qty = st.number_input("Total Quantity Purchased", min_value=1, value=25, help="Number of items purchased to date")
# tenure = st.number_input("Customer Tenure (days)", min_value=1, value=365, help="How long the customer has been with the company")

# # Feature Engineering
# avg_qty_per_order = total_qty / frequency
# log_recency = np.log1p(recency)
# log_frequency = np.log1p(frequency)
# log_monetary = np.log1p(monetary)

# # Prediction Trigger
# if st.button("📈 Generate Insights"):
#     # Predict cluster
#     user_rfm = np.array([[log_recency, log_frequency, log_monetary]])
#     cluster = kmeans.predict(user_rfm)[0]

#     # Predict monetary value
#     input_df = pd.DataFrame({
#         'Recency': [recency],
#         'Frequency': [frequency],
#         'TotalQuantity': [total_qty],
#         'AvgQuantityPerOrder': [avg_qty_per_order],
#         'Tenure': [tenure]
#     })
#     predicted_monetary = rf_model.predict(input_df)[0]

#     # Display Results
#     st.subheader("🎯 Predicted Segment & Value")
    
#     segment_map = {
#         0: "🟣 New/Moderate Buyers",
#         1: "🔴 At-Risk",
#         2: "🟠 High Potential",
#         3: "🟢 VIPs"
#     }

#     descriptions = {
#         0: "These customers show moderate engagement. Nurture with consistent marketing.",
#         1: "These customers may be slipping away. Consider re-engagement campaigns.",
#         2: "These are promising leads. Focused promotion may convert them to loyal buyers.",
#         3: "Top-tier loyal customers. Prioritize retention and premium experiences."
#     }

#     st.success(f"🧩 **Customer Segment:** {segment_map.get(cluster)}")
#     st.info(descriptions.get(cluster))

#     st.metric("💰 Predicted Lifetime Value (CLV)", f"${predicted_monetary:,.2f}")

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: grey;'>"
#     "Developed by <b>Shaney Ang Tech</b> | Powered by Streamlit | © 2025"
#     "</div>",
#     unsafe_allow_html=True
# )

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

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject the sidebar-only CSS
local_css("style.css")

st.title("📊 Customer Segmentation & Lifetime Value Estimator")
st.markdown("This tool classifies your customers into strategic segments and estimates their Customer Lifetime Value (CLV) based on transaction patterns.")

# ------------------ Sidebar Inputs ---------------------
st.sidebar.header("🛒 Customer Purchase Details")

prev_date = st.sidebar.date_input("Previous Purchase Date", value=datetime(2024, 1, 1))
current_date = st.sidebar.date_input("Most Recent Purchase Date", value=datetime(2025, 1, 1))

recency = (current_date - prev_date).days
st.sidebar.markdown(f"📆 **Recency (days since last purchase):** `{recency}`")

frequency = st.sidebar.number_input("Purchase Frequency", min_value=1, value=5, help="Number of orders placed to date")
monetary = st.sidebar.number_input("Total Monetary Spend ($)", min_value=1.0, value=500.0, help="Total spend to date")
total_qty = st.sidebar.number_input("Total Quantity Purchased", min_value=1, value=25, help="Number of items purchased to date")
tenure = st.sidebar.number_input("Customer Tenure (days)", min_value=1, value=365, help="How long the customer has been with the company")

# Feature Engineering
avg_qty_per_order = total_qty / frequency
log_recency = np.log1p(recency)
log_frequency = np.log1p(frequency)
log_monetary = np.log1p(monetary)

# -------------------- Prediction Trigger -------------------
if st.sidebar.button("📈 Generate Insights"):
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

    # Display Results in main area
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
        3: "Top-tier loyal customers. Prioritize retention"
    }
    st.success(f"🧩 **Customer Segment:** {segment_map.get(cluster)}")
    st.info(descriptions.get(cluster))

    st.metric("💰 Predicted Lifetime Value (CLV)", f"${predicted_monetary:,.2f}")

# ---------------- Footer -----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Developed by <b>Shaney Ang Tech</b> | Powered by Streamlit | © 2025"
    "</div>",
    unsafe_allow_html=True
)
