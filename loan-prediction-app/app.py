import joblib
import streamlit as st
import pandas as pd

## Load trained model
model = joblib.load("best_model.pkl")

## Streamlit app
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details below to predict whether the loan will be approved or rejected.")

## User inputs — matches the 5 top features used by the final model
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
loan_term = st.slider("Loan Term (years)", min_value=2, max_value=20, value=10)
loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, max_value=50000000, value=10000000, step=100000)
income_annum = st.number_input("Applicant's Annual Income ($)", min_value=0, max_value=20000000, value=5000000, step=100000)
bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, max_value=30000000, value=4000000, step=100000)

## Predict button
if st.button("Predict Loan Status"):

    ## Convert input data to a DataFrame in the same column order the model was trained on
    df_input = pd.DataFrame({
        'cibil_score': [cibil_score],
        'loan_term': [loan_term],
        'loan_amount': [loan_amount],
        'income_annum': [income_annum],
        'bank_asset_value': [bank_asset_value]
    })

    ## Reindex to be safe, in case column order differs
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    ## Predict
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.success("Prediction: Loan Approved ✅")
    else:
        st.error("Prediction: Loan Rejected ❌")
