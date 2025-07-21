import streamlit as st
import numpy as np
from joblib import load

# Load trained model and scaler
model = load("random_forest_model.joblib")
scaler = load("scaler.pkl") 

# Page settings
st.set_page_config(page_title="Income Prediction", layout="centered", page_icon="💰")
st.title("💼 Income Classification App")
st.markdown("Predict whether an individual's income is **more than $50K/year** based on their demographic and work details.")
st.markdown("---")

# ------------------ Categorical Encodings ------------------ #
workclass_map = {"Private": 2, "Self-emp-not-inc": 3, "Local-gov": 1, "State-gov": 4,
    "Self-emp-inc": 5, "Federal-gov": 0, "Without-pay": 6, "Never-worked": 7}

education_map = {"Preschool": 0, "1st-4th": 1, "5th-6th": 2, "7th-8th": 3,
    "9th": 4, "10th": 5, "11th": 6, "12th": 7, "HS-grad": 8, "Some-college": 9,
    "Assoc-acdm": 10, "Assoc-voc": 11, "Bachelors": 12, "Masters": 13,
    "Doctorate": 14, "Graduate": 15}

education_num_map = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-acdm": 11, "Assoc-voc": 12, "Bachelors": 13, "Masters": 14,
    "Doctorate": 16, "Graduate": 15}

marital_map = {"Divorced": 0, "Married-civ-spouse": 1, "Never-married": 2,
    "Married-spouse-absent": 3, "Separated": 4, "Widowed": 5, "Married-AF-spouse": 6}

occupation_map = {"Adm-clerical": 0, "Armed-Forces": 1, "Farming-fishing": 2,
    "Unknown": 3, "Craft-repair": 4, "Exec-managerial": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Priv-house-serv": 8, "Other-service": 9,
    "Prof-specialty": 10, "Sales": 11, "Tech-support": 12, "Transport-moving": 13,
    "Protective-serv": 14}

race_map = {"White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4}
gender_map = {"Male": 0, "Female": 1}

country_map = {"United-States": 0, "Mexico": 1, "Philippines": 2, "Germany": 3,
    "Canada": 4, "El-Salvador": 5, "India": 6, "Cuba": 7, "England": 8, "Jamaica": 9,
    "South": 10, "China": 11, "Italy": 12, "Dominican-Republic": 13, "Vietnam": 14,
    "Guatemala": 15, "Japan": 16, "Poland": 17, "Columbia": 18, "Taiwan": 19,
    "Haiti": 20, "Iran": 21, "Portugal": 22, "Nicaragua": 23, "Peru": 24,
    "Greece": 25, "France": 26, "Holand-Netherlands": 27, "Thailand": 28,
    "Hong": 29, "Ireland": 30, "Cambodia": 31, "Trinadad&Tobago": 32,
    "Ecuador": 33, "Laos": 34, "Scotland": 35, "Yugoslavia": 36, "Hungary": 37,
    "Outlying-US(Guam-USVI-etc)": 38, "Honduras": 39, "Puerto-Rico": 40, "Unknown": 41}

# --------------------- Input Form --------------------- #
with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=30)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, step=5000, value=200000)
        education = st.selectbox("Education", list(education_map.keys()))
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
        capital_gain = st.number_input("Capital Gain", value=0)
        capital_loss = st.number_input("Capital Loss", value=0)

    with col2:
        workclass = st.selectbox("Workclass", list(workclass_map.keys()))
        marital = st.selectbox("Marital Status", list(marital_map.keys()))
        occupation = st.selectbox("Occupation", list(occupation_map.keys()))
        race = st.selectbox("Race", list(race_map.keys()))
        gender = st.selectbox("Gender", list(gender_map.keys()))
        country = st.selectbox("Native Country", list(country_map.keys()))

    submitted = st.form_submit_button("🔍 Predict Income")

    if submitted:
        # Encode features
        input_features = np.array([[ 
            age,
            fnlwgt,
            education_num_map[education],
            capital_gain,
            capital_loss,
            hours_per_week,
            workclass_map[workclass],
            float(education_map[education]),
            marital_map[marital],
            occupation_map[occupation],
            race_map[race],
            gender_map[gender],
            country_map[country]
        ]])

        # Apply scaling
        input_scaled = scaler.transform(input_features)

        # Show input for debugging
        st.write("🧪 Encoded Input:", input_features)
        st.write("📉 Scaled Input:", input_scaled)

        # Prediction
        y_pred = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)

        result = "> $50K" if y_pred[0] == 1 else "≤ $50K"

        st.markdown("---")
        if result == "> $50K":
            st.success(f"🤑 The model predicts: **more than $50K/year**")
        else:
            st.warning(f"💼 The model predicts: **$50K/year or less**")





