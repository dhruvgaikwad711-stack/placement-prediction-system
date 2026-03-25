import pandas as pd
import random
import sqlite3

conn = sqlite3.connect("database/placement.db")
cursor = conn.cursor()

# Reset table
cursor.execute("DROP TABLE IF EXISTS students")

cursor.execute("""
CREATE TABLE students (
    student_id INTEGER,
    cgpa REAL,
    internships INTEGER,
    projects INTEGER,
    aptitude_score INTEGER,
    communication_skills INTEGER,
    placement TEXT
)
""")

n = 10000

data = []

for i in range(n):
    student_id = i + 1
    cgpa = round(random.uniform(5.0, 9.8), 2)
    internships = random.randint(0, 3)
    projects = random.randint(1, 5)
    aptitude_score = random.randint(30, 95)
    communication_skills = random.randint(4, 10)

    score = cgpa + internships + projects + (aptitude_score / 20) + communication_skills

    placement = "Placed" if score > 18 else "Not Placed"

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

cursor.executemany("INSERT INTO students VALUES (?,?,?,?,?,?,?)", data)

conn.commit()
conn.close()

print(" Fresh dataset generated (10000 rows)")
