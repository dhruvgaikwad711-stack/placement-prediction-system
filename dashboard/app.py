import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Placement Prediction Dashboard",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Placement Prediction System")
st.markdown(
"""
This dashboard analyzes student placement data and predicts  
whether a student will be placed based on academic performance.
"""
)

# -----------------------------
# Load Data FROM SQL
# -----------------------------
conn = sqlite3.connect("database/placement.db")

df = pd.read_sql(
    "SELECT * FROM students",
    conn
)

conn.close()

# -----------------------------
# Metrics
# -----------------------------
col1, col2 = st.columns(2)

placed_count = df[df["placement"]=="Placed"].shape[0]
not_placed_count = df[df["placement"]=="Not Placed"].shape[0]

col1.metric("Placed Students", placed_count)
col2.metric("Not Placed Students", not_placed_count)

st.divider()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Graphs Section
# -----------------------------
st.subheader("📈 Placement Analysis")

col3, col4 = st.columns(2)

# Pie Chart
with col3:
    fig1, ax1 = plt.subplots()
    df["placement"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

# CGPA vs Placement
with col4:
    fig2, ax2 = plt.subplots()
    df.boxplot(column="cgpa", by="placement", ax=ax2)
    st.pyplot(fig2)

st.divider()

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Placement Prediction")

model = pickle.load(
    open("models/placement_model.pkl","rb")
)

col5, col6 = st.columns(2)

with col5:
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.0)
    internships = st.slider("Internships", 0, 3, 1)
    projects = st.slider("Projects", 1, 5, 2)

with col6:
    aptitude = st.slider("Aptitude Score", 30, 100, 60)
    communication = st.slider(
        "Communication Skills", 1, 10, 5
    )

if st.button("Predict Placement"):

    input_data = np.array([[
        cgpa,
        internships,
        projects,
        aptitude,
        communication
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will be Placed")
    else:
        st.error("❌ Student will NOT be Placed")
