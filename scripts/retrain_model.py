import pandas as pd
import sqlite3
import pickle
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Connect to SQL Database
# -----------------------------
conn = sqlite3.connect("database/placement.db")

# Fetch data from students table
df = pd.read_sql(
    "SELECT * FROM students",
    conn
)

conn.close()

# -----------------------------
# Prepare Features & Target
# -----------------------------
X = df.drop(["student_id","placement"], axis=1)

y = df["placement"].map({
    "Not Placed": 0,
    "Placed": 1
})

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier()

model.fit(X, y)

# -----------------------------
# Save Model File
# -----------------------------
pickle.dump(
    model,
    open("models/placement_model.pkl","wb")
)

print("Model trained from SQL database successfully!")

