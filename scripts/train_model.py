import pandas as pd
import sqlite3
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Connect to SQL Database
# -----------------------------
conn = sqlite3.connect("database/placement.db")

# Fetch data from SQL table
df = pd.read_sql("SELECT * FROM students", conn)

conn.close()

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop(["student_id","placement"], axis=1)
y = df["placement"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# -----------------------------
# Train Models
# -----------------------------

rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))

# -----------------------------
# Save Best Model (Random Forest)
# -----------------------------

pickle.dump(
    rf_model,
    open("models/placement_model.pkl","wb")
)

print("\nModel trained & saved successfully!")