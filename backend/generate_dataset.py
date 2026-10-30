import pandas as pd
import numpy as np

# Reproducible random values
np.random.seed(42)

# Number of student profiles
NUM_STUDENTS = 300

data = []

for _ in range(NUM_STUDENTS):

    # Generate varied student profiles
    cgpa = round(np.random.uniform(5.5, 10.0), 2)
    dsa_problems = np.random.randint(0, 301)
    projects = np.random.randint(0, 7)

    # Calculate contributions to the readiness score
    cgpa_score = ((cgpa - 5.0) / 5.0) * 35
    dsa_score = (min(dsa_problems, 200) / 200) * 30
    projects_score = (min(projects, 4) / 4) * 35

    # Add small variation to make the dataset less perfectly predictable
    noise = np.random.normal(0, 3)

    # Final score between 0 and 100
    readiness_score = np.clip(
        cgpa_score + dsa_score + projects_score + noise,
        0,
        100
    )

    data.append({
        "CGPA": cgpa,
        "DSA_Problems": dsa_problems,
        "Projects": projects,
        "Internship_Readiness_Score": round(readiness_score, 2)
    })

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset
df.to_csv("students_regression.csv", index=False)

print("Dataset created successfully!")
print("Total student profiles:", len(df))

print("\nFirst 10 records:")
print(df.head(10))

print("\nDataset information:")
print(df.describe())

print("\nSaved as students_regression.csv")