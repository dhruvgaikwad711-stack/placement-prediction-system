import pickle
import numpy as np

model = pickle.load(open(
    "models/placement_model.pkl","rb"
))

# Example input
cgpa = 8.1
internships = 2
projects = 3
aptitude = 75
communication = 8

input_data = np.array([[
    cgpa, internships, projects,
    aptitude, communication
]])

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Placed")
else:
    print("Not Placed")
