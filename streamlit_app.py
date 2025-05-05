import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# Load trained XGBoost model from JSON
model = xgb.XGBClassifier()
model.load_model('xgb_stroke_model.json')

# App title
st.title("🧠 Adeyinka Stroke Risk Prediction")
st.markdown("Predict your risk of stroke and understand the contributing factors!")

st.markdown("for professional use only")

# User input form
with st.form(key='user_input_form'):

    st.header("👨‍👩‍👧‍👦 Section 1: Demographic & Genetic Data")
    with st.expander("Expand to input Demographic Details"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age (years)', min_value=0, max_value=120, value=50)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            family_history_diabetes = st.selectbox('Family history of diabetes', ['No', 'Yes'])

        with col2:
            af = st.selectbox('Atrial Fibrillation', ['No', 'Yes'])
            smoking_status = st.selectbox('Smoking status', ['Never', 'Ever', 'Current'])
            drinking_status = st.selectbox('Drinking status', ['Never', 'Ever', 'Current'])

    st.header("🏥 Section 2: Clinical & Lifestyle Measurements")
    with st.expander("Expand to input Clinical Measurements"):
        col3, col4 = st.columns(2)

        with col3:
            height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
            weight = st.number_input('Weight (kg)', min_value=20, max_value=200, value=70)
            sbp = st.number_input('Systolic BP (mmHg)', min_value=50, max_value=250, value=120)
            dbp = st.number_input('Diastolic BP (mmHg)', min_value=30, max_value=150, value=80)
            fpg = st.number_input('Fasting Plasma Glucose (mmol/L)', value=5.5)
            fpg_final = st.number_input('FPG at final visit (mmol/L)', value=5.8)

        with col4:
            cholesterol = st.number_input('Cholesterol (mmol/L)', value=4.5)
            triglyceride = st.number_input('Triglyceride (mmol/L)', value=1.5)
            hdl_c = st.number_input('HDL-c (mmol/L)', value=1.2)
            ldl = st.number_input('LDL (mmol/L)', value=2.5)
            alt = st.number_input('ALT (U/L)', value=25.0)
            ast = st.number_input('AST (U/L)', value=22.0)
            bun = st.number_input('BUN (mmol/L)', value=5.0)
            ccr = st.number_input('CCR (umol/L)', value=70.0)
            tyg_index = st.number_input('TyG Index', value=8.5)

        diabetes_during_followup = st.selectbox('Diabetes diagnosed during follow-up?', ['No', 'Yes'])

    # Submit button
    submit_button = st.form_submit_button(label='Predict Stroke Risk')

# Mapping and prediction
if submit_button:
    st.markdown("---")

    # Map categorical inputs
    gender_map = {'Male': 1, 'Female': 2}
    diabetes_map = {'No': 0, 'Yes': 1}
    smoking_map = {'Current': 1, 'Ever': 2, 'Never': 3}
    drinking_map = {'Current': 1, 'Ever': 2, 'Never': 3}
    family_map = {'No': 0, 'Yes': 1}
    af_map = {'No': 0, 'Yes': 1}

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map[gender]],
        'Height': [height],
        'Weight': [weight],
        'BMI': [weight / (height / 100) ** 2],
        'SBP': [sbp],
        'DBP': [dbp],
        'FPG': [fpg],
        'Cholesterol': [cholesterol],
        'Triglyceride': [triglyceride],
        'HDL-c': [hdl_c],
        'LDL': [ldl],
        'ALT': [alt],
        'AST': [ast],
        'BUN': [bun],
        'CCR': [ccr],
        'FPG_of_final_visit': [fpg_final],
        'Diabetes_diagnosed_during_follow-up': [diabetes_map[diabetes_during_followup]],
        'Censor_of_diabetes_at_follow-up': [0],
        'Year_of_follow-up': [5],
        'Smoking_status': [smoking_map[smoking_status]],
        'Drinking_status': [drinking_map[drinking_status]],
        'Family_history_of_diabetes': [family_map[family_history_diabetes]],
        'Atrial_Fibrillation': [af_map[af]],
        'TyG_Index': [tyg_index],
    })

    # Predict
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = int(prediction_proba >= 0.5)

    # Display result
    if prediction:
        st.error(f"⚠️ High risk of stroke! (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Low risk of stroke. (Probability: {prediction_proba:.2f})")

    st.markdown("---")
    st.subheader("🔎 Model Explanation (SHAP Values)")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                          base_values=explainer.expected_value, 
                                          data=input_data.iloc[0]), show=False)
    st.pyplot(fig, bbox_inches='tight')
