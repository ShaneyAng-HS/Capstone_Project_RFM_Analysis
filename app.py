import streamlit as st
import numpy as np
#import pickle
import math



# 2️⃣ Page setup
st.set_page_config(page_title="RFM Cluster Predictor", layout="centered", page_icon="📊")
st.title("🧠 Customer Segmentation with RFM + KMeans")
st.markdown("Input your Recency, Frequency, and Monetary values to find out your cluster.")

# 3️⃣ User input
# recency = st.number_input("Recency (days since last purchase):", min_value=0, max_value=1000, value=100)
# frequency = st.number_input("Frequency (number of purchases):", min_value=1, max_value=1000, value=10)
# monetary = st.number_input("Monetary Value (total spent):", min_value=1.0, value=500.0)

# # 4️⃣ Log-transform the inputs using log1p (log(1 + x))
# log_recency = np.log1p(recency)
# log_frequency = np.log1p(frequency)
# log_monetary = np.log1p(monetary)

# # 5️⃣ Predict cluster
# if st.button("Predict Cluster"):
#     user_rfm = np.array([[log_recency, log_frequency, log_monetary]])
#     cluster = kmeans.predict(user_rfm)[0]
    
#     st.success(f"✅ This customer belongs to **Cluster {cluster}**.")

#     # Optional: Interpret the cluster
#     cluster_notes = {
#         0: "🟣 New/Moderate Buyers.",
#         1: "🔵 At-Risk customers.",
#         2: "🟠 High Potential customers.",
#         3: "🟢 VIPs."
#     }
#     st.markdown(cluster_notes.get(cluster, "ℹ️ No specific description for this cluster."))

