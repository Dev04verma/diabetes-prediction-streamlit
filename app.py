import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests



# Load model and scaler
model, scaler = pickle.load(open("model.pkl", "rb"))

# Custom style
st.set_page_config(page_title="💉 Diabetes Prediction App", layout="wide")
# Load animation from Lottie

# Doctor-style dark theme CSS
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .main {
        background-color: #121212;
    }
    input, select, textarea {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-radius: 5px;
        padding: 5px;
    }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stDownloadButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    h1, h2, h3, h4 {
        color: #00b0ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 10px;
        border-radius: 10px;
        color: #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💉 Diabetes Prediction App")
st.markdown("Welcome Doctor! Use this tool to **predict diabetes**, **analyze patient data**, and **visualize risk indicators**.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Predict One", "📂 Predict From File", "📊 Visual Insights"])

# 🔹 Predict One
with tab1:
    st.subheader("👤 Enter Patient Details:")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0.0)
        glucose = st.number_input("Glucose", 0.0)
        blood_pressure = st.number_input("Blood Pressure", 0.0)
        skin_thickness = st.number_input("Skin Thickness", 0.0)
    with col2:
        insulin = st.number_input("Insulin", 0.0)
        bmi = st.number_input("BMI", 0.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0)
        age = st.number_input("Age", 0.0)

    if st.button("🧮 Predict Now"):
        inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaled = scaler.transform(inputs)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.error("⚠️ The patient is likely **Diabetic**.")
        else:
            st.success("✅ The patient is likely **Not Diabetic**.")

# 🔹 Predict From File
with tab2:
    st.subheader("📂 Upload CSV File for Multiple Patients")
    file = st.file_uploader("Upload your file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("✅ File Loaded Successfully!")
        st.dataframe(df.head())

        try:
            X = df.values
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in y_pred]

            st.success("🔍 Predictions Completed!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results", data=csv, file_name="predicted_results.csv", mime='text/csv')
        except:
            st.error("❌ Error! Please check column order.")

# 🔹 Custom Visualizations
with tab3:
    st.subheader("📊 Patient Data Insights (using PIMA Dataset)")
    df = pd.read_csv("diabetes.csv")

    st.markdown("📈 Outcome Distribution Chart")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Outcome", palette="Set2", ax=ax1)
    ax1.set_xticklabels(['Not Diabetic', 'Diabetic'])
    st.pyplot(fig1)

    st.markdown("Age vs Diabetes")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="Age", hue="Outcome", multiple="stack", palette="Set1", bins=20, ax=ax2)
    st.pyplot(fig2)

    st.markdown("BMI Distribution by Outcome")
    fig3, ax3 = plt.subplots()
    sns.kdeplot(data=df, x="BMI", hue="Outcome", fill=True, ax=ax3)
    st.pyplot(fig3)

    st.markdown("🧪 Glucose Level Distribution")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x="Outcome", y="Glucose", palette="coolwarm", ax=ax4)
    ax4.set_xticklabels(['Not Diabetic', 'Diabetic'])
    st.pyplot(fig4)

    st.markdown("🧠 Custom Feature vs Outcome")
    feature = st.selectbox("Select Feature:", df.columns[:-1])
    fig5, ax5 = plt.subplots()
    sns.histplot(data=df, x=feature, hue="Outcome", multiple="stack", ax=ax5)
    st.pyplot(fig5)
