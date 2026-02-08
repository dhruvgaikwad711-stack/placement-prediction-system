import pandas as pd
import sqlite3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Connect to SQL Database
# -----------------------------
conn = sqlite3.connect("database/placement.db")

# Fetch data from SQL table
df = pd.read_sql(
    "SELECT * FROM students",
    conn
)

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

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Model Accuracy:", acc)

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(
    model,
    open("models/placement_model.pkl","wb")
)

print("Model trained & saved from SQL data!")
