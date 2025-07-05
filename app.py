import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model and scaler from file
model, scaler = pickle.load(open("model.pkl", "rb"))

# Streamlit page config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Title
st.title("💉 Diabetes Prediction App")
st.write("This app predicts whether a person is likely to have **diabetes** using medical data.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Predict One", "📂 Predict From File", "📊 Data Visualization"])

# -------------------------------------
# 🔹 Tab 1: Single Patient Prediction
with tab1:
    st.subheader("Enter Patient Details:")

    name = st.text_input("Enter Patient Name")

    pregnancies = st.number_input("Pregnancies", 0.0)
    glucose = st.number_input("Glucose", 0.0)
    blood_pressure = st.number_input("Blood Pressure", 0.0)
    skin_thickness = st.number_input("Skin Thickness", 0.0)
    insulin = st.number_input("Insulin", 0.0)
    bmi = st.number_input("BMI", 0.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0.0)

    if st.button("Predict Now"):
        if name.strip() == "":
            st.warning("⚠️ Please enter the patient's name.")
        else:
            inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            scaled = scaler.transform(inputs)
            prediction = model.predict(scaled)[0]

            # Prediction result
            if prediction == 1:
                st.error(f"🚨 {name}, you are likely **Diabetic**.")
            else:
                st.success(f"✅ {name}, you are likely **Not Diabetic**.")

            # Chart showing prediction
            st.subheader("📊 Prediction Summary")
            categories = ['Not Diabetic', 'Diabetic']
            values = [1 if prediction == 0 else 0, 1 if prediction == 1 else 0]

            fig, ax = plt.subplots()
            sns.barplot(x=categories, y=values,
                        palette=["green" if prediction == 0 else "lightgray", "red" if prediction == 1 else "lightgray"],
                        ax=ax)
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Prediction Confidence")
            ax.set_title(f"Prediction Result for {name}")

            for i, v in enumerate(values):
                ax.text(i, v + 0.05, str(v), color='black', ha='center', fontweight='bold')

            st.pyplot(fig)

# -------------------------------------
# 🔹 Tab 2: Predict from CSV File
with tab2:
    st.subheader("Upload CSV File for Bulk Prediction")
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)
            st.success("✅ File Loaded Successfully.")
            st.dataframe(df.head())

            X = df.values
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in y_pred]

            st.success("✅ Predictions completed:")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results CSV", data=csv, file_name="diabetes_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error("❌ Error: Please check CSV format and column order.")

# -------------------------------------
# 🔹 Tab 3: Visualization
with tab3:
    st.subheader("📊 Data Visualizations")
    use_sample = st.checkbox("Use Sample Dataset", value=True)

    if use_sample:
        df = pd.read_csv("diabetes.csv")
        st.dataframe(df.head())

        # Outcome Distribution
        st.markdown("📈 Outcome Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Outcome', data=df, ax=ax1)
        ax1.set_xticklabels(['Not Diabetic (0)', 'Diabetic (1)'])
        st.pyplot(fig1)

        # # Correlation Heatmap
        # st.markdown("### 🔥 Feature Correlation Heatmap")
        # fig2, ax2 = plt.subplots(figsize=(7, 4))
        # sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        # st.pyplot(fig2)
        
        #feature vs outcome
        st.markdown("🧠 Custom Feature vs Outcome")
        feature = st.selectbox("Select Feature:", df.columns[:-1])
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=feature, hue="Outcome", multiple="stack", ax=ax5)
        st.pyplot(fig5)
        

        # Glucose Histogram
        st.markdown("🧪 Glucose Level Distribution")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Glucose'], kde=True, bins=30, ax=ax3)
        st.pyplot(fig3)
