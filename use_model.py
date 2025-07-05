import pickle
import numpy as np

# Load the saved model and scaler
model, scaler = pickle.load(open("model.pkl", "rb"))

# Sample patient input (you can replace this with your own)
sample_data = np.array([[2, 130, 80, 20, 100, 28.0, 0.5, 35]])  # 8 features

# Scale the input
scaled_input = scaler.transform(sample_data)

# Predict
prediction = model.predict(scaled_input)[0]

# Print result
if prediction == 1:
    print("🚨 The person is likely Diabetic.")
else:
    print("✅ The person is likely Not Diabetic.")
