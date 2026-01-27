# ===============================
# CREDIT SCORING MODEL - STREAMLIT UI
# Logistic Regression, Random Forest & XGBoost
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Page config
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 6rem;
        font-weight: bold;
        color: #28A745;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-good {
        background-color: #D4EDDA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        color: #155724 !important;
    }
    .prediction-good h2 {
        color: #155724 !important;
        margin: 0 0 10px 0;
    }
    .prediction-good p {
        color: #155724 !important;
        margin: 5px 0;
    }
    .prediction-medium {
        background-color: #FFF3CD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
        color: #856404 !important;
    }
    .prediction-medium h2 {
        color: #856404 !important;
        margin: 0 0 10px 0;
    }
    .prediction-medium p {
        color: #856404 !important;
        margin: 5px 0;
    }
    .prediction-bad {
        background-color: #F8D7DA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #DC3545;
        color: #721C24 !important;
    }
    .prediction-bad h2 {
        color: #721C24 !important;
        margin: 0 0 10px 0;
    }
    .prediction-bad p {
        color: #721C24 !important;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD AND TRAIN MODELS
# ===============================

@st.cache_resource
def load_and_train_model():
    # 1. Load the dataset
    df = pd.read_csv("Dataset/german.data-numeric", sep=r'\s+', header=None)
    
    # Assign column names (24 features + 1 target)
    df.columns = [f'Feature_{i}' for i in range(1, 25)] + ['target']
    
    # 2. Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Drop features 21-24 (only keep 1-20)
    X = X.drop(['Feature_21', 'Feature_22', 'Feature_23', 'Feature_24'], axis=1)
    
    # Convert target: 1 -> 0 (good), 2 -> 1 (bad)
    y = y.replace({1: 0, 2: 1})
    
    # Store feature names for input form
    feature_names = X.columns.tolist()
    
    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Feature scaling (needed for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===============================
    # MODEL 1: Logistic Regression
    # ===============================
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    
    log_predictions = log_model.predict(X_test_scaled)
    log_accuracy = accuracy_score(y_test, log_predictions)
    
    # ===============================
    # MODEL 2: Random Forest
    # ===============================
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    # ===============================
    # MODEL 3: XGBoost
    # ===============================
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    
    # ===============================
    # SAVE MODELS
    # ===============================
    import joblib
    import os
    
    # Create models folder if not exists
    os.makedirs('models', exist_ok=True)
    
    # Save all models and scaler
    joblib.dump(log_model, 'models/logistic_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return log_model, rf_model, xgb_model, scaler, log_accuracy, rf_accuracy, xgb_accuracy, feature_names

# Load models
log_model, rf_model, xgb_model, scaler, log_accuracy, rf_accuracy, xgb_accuracy, feature_names = load_and_train_model()

# ===============================
# STREAMLIT UI
# ===============================

# Header
st.markdown('<p class="main-header">Credit Scoring Model</p>', unsafe_allow_html=True)
st.markdown("### Predict credit risk using Machine Learning")

# Sidebar - Model Selection
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["Logistic Regression", "Random Forest", "XGBoost"],
    index=2
)

# Show model accuracies in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Logistic Regression", f"{log_accuracy:.1%}")
st.sidebar.metric("Random Forest", f"{rf_accuracy:.1%}")
st.sidebar.metric("XGBoost", f"{xgb_accuracy:.1%}")

# ===============================
# INPUT FORM
# ===============================

st.markdown("---")
st.markdown("### Enter Customer Information")

# Create input columns
col1, col2, col3 = st.columns(3)

# Column 1
with col1:
    # Feature 1: Account Status (Dropdown)
    account_status_options = {
        "< 0 DM": 1,
        "0 - 200 DM": 2,
        ">= 200 DM": 3,
        "No checking account": 4
    }
    account_status = st.selectbox("Account Status", list(account_status_options.keys()))
    feature_1 = account_status_options[account_status]
    
    # Feature 4: Purpose (Dropdown)
    purpose_options = {
        "Car (new)": 0,
        "Car (used)": 1,
        "Furniture": 2,
        "Radio/TV": 3,
        "Domestic appliances": 4,
        "Repairs": 5,
        "Education": 6,
        "Vacation": 7,
        "Retraining": 8,
        "Business": 9,
        "Other": 10
    }
    purpose = st.selectbox("Purpose", list(purpose_options.keys()))
    feature_4 = purpose_options[purpose]
    
    # Feature 7: Employment Since (Dropdown)
    employment_options = {
        "Unemployed": 1,
        "< 1 year": 2,
        "1 - 4 years": 3,
        "4 - 7 years": 4,
        ">= 7 years": 5
    }
    employment = st.selectbox("Employment Since", list(employment_options.keys()))
    feature_7 = employment_options[employment]
    
    # Feature 10: Other Debtors (Dropdown)
    other_debtors_options = {
        "None": 1,
        "Co-applicant": 2,
        "Guarantor": 3
    }
    other_debtors = st.selectbox("Other Debtors", list(other_debtors_options.keys()))
    feature_10 = other_debtors_options[other_debtors]
    
    # Feature 13: Age (Number input)
    feature_13 = st.number_input("Age (years)", min_value=18, max_value=75, value=35)
    
    # Feature 16: Existing Credits (Slider)
    feature_16 = st.slider("Existing Credits", min_value=1, max_value=4, value=1)
    
    # Feature 19: Telephone (Toggle)
    telephone = st.toggle("Telephone Registered", value=True)
    feature_19 = 2 if telephone else 1

# Column 2
with col2:
    # Feature 2: Duration (Slider)
    feature_2 = st.slider("Duration (months)", min_value=6, max_value=72, value=24)
    
    # Feature 5: Credit Amount (Number input)
    feature_5 = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=2500)
    
    # Feature 8: Installment Rate (Slider)
    feature_8 = st.slider("Installment Rate (% of income)", min_value=1, max_value=4, value=2)
    
    # Feature 11: Residence Since (Slider)
    feature_11 = st.slider("Residence Since (years)", min_value=1, max_value=4, value=2)
    
    # Feature 14: Other Plans (Dropdown)
    other_plans_options = {
        "Bank": 1,
        "Stores": 2,
        "None": 3
    }
    other_plans = st.selectbox("Other Installment Plans", list(other_plans_options.keys()))
    feature_14 = other_plans_options[other_plans]
    
    # Feature 17: Job Type (Dropdown)
    job_options = {
        "Unemployed / Unskilled (non-resident)": 1,
        "Unskilled (resident)": 2,
        "Skilled employee": 3,
        "Highly skilled / Self-employed": 4
    }
    job = st.selectbox("Job Type", list(job_options.keys()))
    feature_17 = job_options[job]
    
    # Feature 20: Foreign Worker (Toggle)
    foreign_worker = st.toggle("Foreign Worker", value=True)
    feature_20 = 1 if foreign_worker else 2

# Column 3
with col3:
    # Feature 3: Credit History (Dropdown)
    credit_history_options = {
        "No credits taken / all paid": 0,
        "All credits paid on time": 1,
        "Existing credits paid on time": 2,
        "Delay in past payments": 3,
        "Critical account": 4
    }
    credit_history = st.selectbox("Credit History", list(credit_history_options.keys()))
    feature_3 = credit_history_options[credit_history]
    
    # Feature 6: Savings Account (Dropdown)
    savings_options = {
        "No savings / < 100 DM": 1,
        "100 - 500 DM": 2,
        "500 - 1000 DM": 3,
        ">= 1000 DM": 4,
        "Unknown / No savings account": 5
    }
    savings = st.selectbox("Savings Account", list(savings_options.keys()))
    feature_6 = savings_options[savings]
    
    # Feature 9: Personal Status (Dropdown)
    personal_status_options = {
        "Male: Divorced/Separated": 1,
        "Female: Divorced/Separated/Married": 2,
        "Male: Single": 3,
        "Male: Married/Widowed": 4,
        "Female: Single": 5
    }
    personal_status = st.selectbox("Personal Status", list(personal_status_options.keys()))
    feature_9 = personal_status_options[personal_status]
    
    # Feature 12: Property (Dropdown)
    property_options = {
        "Real estate": 1,
        "Savings agreement / Life insurance": 2,
        "Car or other": 3,
        "No property": 4
    }
    property_type = st.selectbox("Property", list(property_options.keys()))
    feature_12 = property_options[property_type]
    
    # Feature 15: Housing (Dropdown)
    housing_options = {
        "Rent": 1,
        "Own": 2,
        "Free": 3
    }
    housing = st.selectbox("Housing", list(housing_options.keys()))
    feature_15 = housing_options[housing]
    
    # Feature 18: Dependents (Slider)
    feature_18 = st.slider("Number of Dependents", min_value=1, max_value=2, value=1)

# ===============================
# PREDICTION
# ===============================

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_btn = st.button("Predict Credit Risk", type="primary", use_container_width=True)

if predict_btn:
    # Prepare input in correct order (Feature_1 to Feature_20)
    input_data = {
        'Feature_1': feature_1,
        'Feature_2': feature_2,
        'Feature_3': feature_3,
        'Feature_4': feature_4,
        'Feature_5': feature_5,
        'Feature_6': feature_6,
        'Feature_7': feature_7,
        'Feature_8': feature_8,
        'Feature_9': feature_9,
        'Feature_10': feature_10,
        'Feature_11': feature_11,
        'Feature_12': feature_12,
        'Feature_13': feature_13,
        'Feature_14': feature_14,
        'Feature_15': feature_15,
        'Feature_16': feature_16,
        'Feature_17': feature_17,
        'Feature_18': feature_18,
        'Feature_19': feature_19,
        'Feature_20': feature_20,
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Select model and make prediction
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        prediction = log_model.predict(input_scaled)[0]
        probability = log_model.predict_proba(input_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_df)[0]
        probability = rf_model.predict_proba(input_df)[0]
    else:  # XGBoost
        prediction = xgb_model.predict(input_df)[0]
        probability = xgb_model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.markdown("### Prediction Result")
    
    result_col1, result_col2 = st.columns(2)
    
    good_prob = probability[0] * 100
    bad_prob = probability[1] * 100
    
    with result_col1:
        # Determine risk level based on probability
        if good_prob >= 60:
            st.markdown("""
            <div class="prediction-good">
                <h2>GOOD CREDIT</h2>
                <p>This customer is predicted to be a <strong>low-risk</strong> borrower.</p>
                <p>Recommendation: Approve the loan application.</p>
            </div>
            """, unsafe_allow_html=True)
        elif good_prob >= 40:
            st.markdown("""
            <div class="prediction-medium">
                <h2>MEDIUM CREDIT</h2>
                <p>This customer is predicted to be a <strong>moderate-risk</strong> borrower.</p>
                <p>Recommendation: Request additional documentation before approval.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-bad">
                <h2>BAD CREDIT</h2>
                <p>This customer is predicted to be a <strong>high-risk</strong> borrower.</p>
                <p>Recommendation: Review carefully before approval.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with result_col2:
        st.markdown("#### Confidence Score")
        
        st.progress(int(good_prob))
        st.write(f"Good Credit Probability: **{good_prob:.1f}%**")
        
        st.progress(int(bad_prob))
        st.write(f"Bad Credit Probability: **{bad_prob:.1f}%**")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Credit Scoring Model | Built with Streamlit | German Credit Dataset</p>
</div>
""", unsafe_allow_html=True)
