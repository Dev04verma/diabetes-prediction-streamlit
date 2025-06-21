import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Sample input (replace with your own values)
sample = np.array([[2, 130, 70, 20, 85, 28.0, 0.5, 40]])

# Scale and predict
scaled_sample = scaler.transform(sample)
prediction = model.predict(scaled_sample)
probability = model.predict_proba(scaled_sample)[0][1]

# Show result
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
print(f"Risk Probability: {probability:.2f}")
