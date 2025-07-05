
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle missing/zero values if needed (optional for real use)
# Example: Replace zeros in some columns with median (if you wish)
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    X[col] = X[col].replace(0, X[col].median())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler together
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained and saved as model.pkl")
