# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load Trained Artifacts ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please run the 'train_model.py' script first.")
    st.stop()

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

def generate_plot_summary(explanation, feature_names_list):
    """Generates a plain-language summary of a SHAP explanation."""
    base_value = explanation.base_values
    final_value = np.sum(explanation.values) + base_value
    
    abs_shap = np.abs(explanation.values)
    max_impact_idx = np.argmax(abs_shap)
    
    most_impactful_feature = feature_names_list[max_impact_idx]
    impact_value = explanation.values[max_impact_idx]
    feature_input_value = explanation.data[max_impact_idx]

    if final_value > base_value:
        trend = "an **above-average** profile for loan approval"
    else:
        trend = "a **below-average** profile for loan approval"

    if impact_value > 0.01:
        direction = "**significantly increased** the chance of approval."
    elif impact_value < -0.01:
        direction = "**significantly decreased** the chance of approval."
    else:
        direction = "had a **negligible impact** on the outcome."
        insight = "\n\n💡 **Insight:** The feature contributions look small because this applicant's profile is very typical."
    
    if most_impactful_feature == 'Credit_History':
        feature_input_value = "Good" if feature_input_value == 1.0 else "Bad"

    summary = (
        f"The model's average prediction (base value) is **{base_value:.2f}**. "
        f"For this applicant, the final score is **{final_value:.2f}**, which indicates {trend}.\n\n"
        f"The most important factor was **{most_impactful_feature}**. "
        f"The applicant's value of **'{feature_input_value}'** {direction}"
    )
    
    if 'insight' in locals():
        summary += insight
        
    return summary

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("💡REAL TIME LOAN ELIGIBILITY ASSESSMENT USING MACHINE LEARNING AND XAI TECHNIQUE")
st.write("""
This app predicts whether a loan application will be **approved or rejected** using a machine learning model.
It also uses **SHAP** to explain the key factors influencing the decision.
""")
st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Applicant Information")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    credit_history = st.sidebar.selectbox('Credit History', (1.0, 0.0), format_func=lambda x: 'Good' if x == 1.0 else 'Bad')
    property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))
    loan_amount = st.sidebar.slider('Loan Amount ($)', 10, 700, 150)
    total_income = st.sidebar.slider('Total Monthly Income ($)', 1500, 81000, 5000)
    loan_amount_term = st.sidebar.slider('Loan Amount Term (Months)', 36, 480, 360)

    loan_amount_log = np.log(loan_amount)
    total_income_log = np.log(total_income)

    data = {
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_employed,
        'Loan_Amount_Term': loan_amount_term, 'Credit_History': credit_history,
        'Property_Area': property_area, 'LoanAmountlog': loan_amount_log,
        'TotalIncomelog': total_income_log
    }

    features = pd.DataFrame(data, index=[0])
    return features[feature_names]

input_df = user_input_features()

st.subheader("Applicant's Input")
st.dataframe(input_df, hide_index=True)

# --- Prediction and SHAP Explanation ---
if st.sidebar.button("Predict Loan Status", type="primary"):
    processed_input = input_df.copy()
    for col_idx, encoder in encoders.items():
        col_name = feature_names[col_idx]
        processed_input[col_name] = encoder.transform(processed_input[col_name])

    scaled_input = scaler.transform(processed_input)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    status = target_encoder.inverse_transform(prediction)[0]
    
    st.subheader("Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if status == 'Y':
            st.metric(label="Loan Status", value="ELIGIBLE ✔️", delta="High Confidence")
            st.balloons() 
        else:
            st.metric(label="Loan Status", value="NOT ELIGIBLE ❌", delta="High Confidence", delta_color="inverse")
            # --- THIS IS THE NEW LINE ---
            st.toast('The application did not meet the required criteria.', icon='😞')
    
    with col2:
        approval_prob = prediction_proba[0][1]
        st.write("Eligibility Probability Score:")
        st.progress(approval_prob)
        st.markdown(f"**{approval_prob:.1%}**")

    st.markdown("---")
    st.subheader("Why did the model decide this? (XAI Explanation)")

    shap_explanation = explainer(scaled_input, check_additivity=False)
    explanation_for_approved = shap_explanation[:,:,1]
    
    plot_summary = generate_plot_summary(explanation_for_approved[0], feature_names)
    st.info(plot_summary)

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Waterfall Plot")
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(explanation_for_approved[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("#### Feature Impact Bar Plot")
        plt.figure(figsize=(8, 6))
        shap.plots.bar(explanation_for_approved[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

else:
    st.info("Adjust the parameters in the sidebar and click 'Predict Loan Status' to get a result.")