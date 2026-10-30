import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv("students_regression.csv")

print("=" * 50)
print("       CAREERFORGE AI")
print("   Internship Readiness Prediction")
print("=" * 50)

print("\nDataset loaded successfully!")
print("Total student profiles:", len(df))

print("\nFirst 5 records:")
print(df.head())




X = df[["CGPA", "DSA_Problems", "Projects"]]

y = df["Internship_Readiness_Score"]

print("\nFeatures:", list(X.columns))
print("Target: Internship_Readiness_Score")




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))




model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)



model.fit(X_train, y_train)

print("\nModel training completed successfully!")



y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("          MODEL EVALUATION")
print("=" * 50)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")




print("\n" + "=" * 50)
print("          FEATURE IMPORTANCE")
print("=" * 50)

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(f"{feature}: {importance:.3f}")




sample_student = pd.DataFrame([{
    "CGPA": 8.2,
    "DSA_Problems": 120,
    "Projects": 3
}])

predicted_score = model.predict(sample_student)[0]

print("\n" + "=" * 50)
print("       SAMPLE STUDENT PROFILE")
print("=" * 50)

print("CGPA: 8.2")
print("DSA problems solved: 120")
print("Projects completed: 3")

print(
    f"\nPredicted Internship Readiness Score: "
    f"{predicted_score:.2f}/100"
)



if predicted_score < 40:
    category = "Needs Significant Improvement"

elif predicted_score < 60:
    category = "Developing"

elif predicted_score < 80:
    category = "Progressing Well"

else:
    category = "Strong Readiness Profile"

print("Readiness Category:", category)



print("\n" + "=" * 50)
print("       PERSONALIZED ACTION PLAN")
print("=" * 50)

if sample_student["CGPA"].iloc[0] < 7.5:
    print("- Focus on strengthening academic performance.")
else:
    print("- Maintain your academic performance.")

if sample_student["DSA_Problems"].iloc[0] < 100:
    print("- Practise DSA consistently and revise core patterns.")
else:
    print("- Continue solving DSA problems and revise previously solved ones.")

if sample_student["Projects"].iloc[0] < 2:
    print("- Build practical projects and document your contributions.")
else:
    print("- Improve existing projects and prepare to explain your technical decisions.")

print("- Practise technical interview questions.")
print("- Strengthen your resume and communication skills.")

print("\n" + "=" * 50)
print("IMPORTANT DISCLAIMER")
print("=" * 50)

print(
    "This is an educational prototype trained on synthetic data."
)
print(
    "The score is NOT a real-world internship selection probability."
)
print(
    "Actual selection depends on many additional factors."
)