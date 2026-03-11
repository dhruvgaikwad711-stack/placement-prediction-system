import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

# connect database
conn = sqlite3.connect("placement.db")

# read data
df = pd.read_sql_query("SELECT * FROM students", conn)

# features and target
X = df.drop("placement", axis=1)
y = df["placement"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)