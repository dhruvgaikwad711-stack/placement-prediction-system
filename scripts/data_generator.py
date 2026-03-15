import pandas as pd
import random
import sqlite3

# Connect to SQL database
conn = sqlite3.connect("database/placement.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER,
    cgpa REAL,
    internships INTEGER,
    projects INTEGER,
    aptitude_score INTEGER,
    communication_skills INTEGER,
    placement TEXT
)
""")

# Number of records
n = 50000

data = []

for i in range(n):
    student_id = i + 1
    cgpa = round(random.uniform(5.0, 9.8), 2)
    internships = random.randint(0, 3)
    projects = random.randint(1, 5)
    aptitude_score = random.randint(30, 95)
    communication_skills = random.randint(4, 10)

    # create score for placement decision
    score = cgpa + internships + projects + (aptitude_score / 20) + communication_skills

    if score > 18:
        placement = "Placed"
    else:
        placement = "Not Placed"

    # add randomness to reduce overfitting
    if random.random() > 0.90:
        placement = "Placed" if placement == "Not Placed" else "Not Placed"

    data.append((
        student_id,
        cgpa,
        internships,
        projects,
        aptitude_score,
        communication_skills,
        placement
    ))

# Insert data into SQL table
cursor.executemany("""
INSERT INTO students VALUES (?,?,?,?,?,?,?)
""", data)

conn.commit()
conn.close()

print("Dataset generated and stored in SQL database successfully!")
