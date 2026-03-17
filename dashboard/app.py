import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
st.title("Student Placement Prediction System")

st.markdown("""
This dashboard analyzes student placement data and predicts  
whether a student will be placed based on academic performance.
""")

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
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filter Data")

min_cgpa = st.sidebar.slider("Minimum CGPA", 5.0, 10.0, 5.0)

min_internships = st.sidebar.selectbox(
    "Minimum Internships",
    [0,1,2,3]
)

placement_filter = st.sidebar.selectbox(
    "Placement Status",
    ["All","Placed","Not Placed"]
)

filtered_df = df[
    (df["cgpa"] >= min_cgpa) &
    (df["internships"] >= min_internships)
]

if placement_filter != "All":
    filtered_df = filtered_df[
        filtered_df["placement"] == placement_filter
    ]

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(filtered_df))

col2.metric(
    "Placed Students",
    filtered_df[filtered_df["placement"]=="Placed"].shape[0]
)

col3.metric(
    "Not Placed Students",
    filtered_df[filtered_df["placement"]=="Not Placed"].shape[0]
)

st.divider()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Explorer")

st.write("Filtered Dataset Size:", len(filtered_df))

sort_column = st.selectbox(
    "Sort Dataset By",
    ["cgpa","internships","aptitude_score"]
)

sorted_df = filtered_df.sort_values(
    by=sort_column,
    ascending=False
)

st.dataframe(sorted_df.head(10))

st.divider()

# -----------------------------
# Placement Analysis
# -----------------------------
st.subheader("Placement Analysis")

col4, col5 = st.columns(2)

# Placement distribution
with col4:

    fig1, ax1 = plt.subplots()

    filtered_df["placement"].value_counts().plot.bar(
        color=["#4CAF50","#FF5252"],
        ax=ax1
    )

    ax1.set_ylabel("Students")

    st.pyplot(fig1)

# CGPA vs placement
with col5:

    fig2, ax2 = plt.subplots()

    filtered_df.boxplot(column="cgpa", by="placement", ax=ax2)

    st.pyplot(fig2)

st.divider()

# -----------------------------
# Feature Analysis
# -----------------------------
col6, col7 = st.columns(2)

with col6:

    st.subheader("CGPA Distribution")

    fig3, ax3 = plt.subplots()

    ax3.hist(filtered_df["cgpa"], bins=20)

    st.pyplot(fig3)

with col7:

    st.subheader("Internships vs Students")

    fig4, ax4 = plt.subplots()

    filtered_df.groupby("internships")["placement"].count().plot.bar(ax=ax4)

    st.pyplot(fig4)

st.divider()

# -----------------------------
# Key Insights
# -----------------------------
st.subheader("Key Insights")

st.write("Average CGPA:", round(filtered_df["cgpa"].mean(),2))

st.write("Average Aptitude Score:", round(filtered_df["aptitude_score"].mean(),2))

st.write("Average Communication Skills:", round(filtered_df["communication_skills"].mean(),2))

st.divider()

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(
    open("models/placement_model.pkl","rb")
)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

importance = model.feature_importances_

features = ["cgpa","internships","projects","aptitude_score","communication_skills"]

importance_df = pd.DataFrame({
    "Feature":features,
    "Importance":importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig_imp, ax_imp = plt.subplots()

ax_imp.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="#2196F3"
)

st.pyplot(fig_imp)

st.divider()

# -----------------------------
# Model Comparison
# -----------------------------
st.subheader("Model Comparison")

X = df.drop(["student_id","placement"], axis=1)

y = df["placement"].map({"Not Placed":0,"Placed":1})

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Logistic regression
log_model = LogisticRegression()

log_model.fit(X_train,y_train)

log_preds = log_model.predict(X_test)

log_acc = accuracy_score(y_test,log_preds)

# Random forest accuracy
rf_preds = model.predict(X_test)

rf_acc = accuracy_score(y_test,rf_preds)

models = ["Random Forest","Logistic Regression"]

scores = [rf_acc,log_acc]

fig_cmp, ax_cmp = plt.subplots()

ax_cmp.bar(models,scores,color=["#4CAF50","#FF9800"])

ax_cmp.set_ylabel("Accuracy")

st.pyplot(fig_cmp)

st.divider()

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Placement Prediction")

col8, col9 = st.columns(2)

with col8:
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.0)
    internships = st.slider("Internships", 0, 3, 1)
    projects = st.slider("Projects", 1, 5, 2)

with col9:
    aptitude = st.slider("Aptitude Score", 30, 100, 60)
    communication = st.slider("Communication Skills", 1, 10, 5)

if st.button("Predict Placement"):

    input_data = np.array([[ 
        cgpa,
        internships,
        projects,
        aptitude,
        communication
    ]])

    # Model prediction
    prediction = model.predict(input_data)

    # Prediction probability
    probability = model.predict_proba(input_data)

    placement_prob = probability[0][1] * 100

    st.subheader("Prediction Result")

    st.write("Placement Probability:", round(placement_prob,2), "%")

    # Progress bar visualization
    st.progress(int(placement_prob))

    if prediction[0] == 1:
        st.success("Student is likely to be PLACED")

    else:
        st.error("Student is NOT likely to be placed")