import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load assets
model = joblib.load("salary_prediction_model.pkl")
model_columns = joblib.load("model_columns.pkl")
X_test, y_test, y_pred = joblib.load("test_predictions.pkl")
original_data = pd.read_csv("Salary Data.csv")
original_data.dropna(inplace=True)

# --- Streamlit UI ---

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1aumxhk {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("💼 Employee Salary Predictor")
    st.markdown("""
    AI-powered salary estimation tool based on:
    - Age
    - Gender
    - Education Level
    - Job Title
    - Years of Experience
    """)
    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.write(f"**Selected Model:** {model.__class__.__name__}")
    st.write(f"**Mean Absolute Error (MAE):** {abs(y_test - y_pred).mean():,.2f}")
    st.write(f"**Mean Squared Error (MSE):** {(y_test - y_pred).pow(2).mean():,.2f}")
    st.write(f"**R-squared (R²):** {model.score(X_test, y_test):.4f}")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Estimate salaries using AI based on employee profile</p><hr>", unsafe_allow_html=True)

st.header("🧾 Enter Employee Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👶 Age", 18, 65, 30)
    gender = st.radio("👤 Gender", ["Male", "Female"])
    education = st.radio("🎓 Education Level", ["Bachelor's", "Master's", "PhD"])

with col2:
    job = st.selectbox("💼 Job Title", original_data['Job Title'].unique())
    exp = st.slider("🕒 Years of Experience", 0, 50, 5)

# Validations
if exp >= age or exp > (age - 20) or age < (exp + 18):
    st.error("Please enter valid age and experience values.")
    st.stop()
if education == "Master's" and age < 23:
    st.error("Age too low for a Master's degree.")
    st.stop()
if education == "PhD" and age < 26:
    st.error("Age too low for a PhD.")
    st.stop()

# Construct input data
input_dict = {col: 0 for col in model_columns}
input_dict["Age"] = age
input_dict["Years of Experience"] = exp

if f"Gender_{gender}" in input_dict:
    input_dict[f"Gender_{gender}"] = 1
if f"Education Level_{education}" in input_dict:
    input_dict[f"Education Level_{education}"] = 1
if f"Job Title_{job}" in input_dict:
    input_dict[f"Job Title_{job}"] = 1

X_input = pd.DataFrame([input_dict])[model_columns]

if st.button("🔍 Predict Salary"):
    salary = model.predict(X_input)[0]
    st.success(f"💰 Estimated Salary: ₹{salary:,.2f}",  icon="📢")
    st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)

# Feature importance
with st.expander("🧠 Feature Importance"):
    coef_df = pd.DataFrame({"Feature": model_columns, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)
    st.bar_chart(coef_df.set_index("Feature"))

# Scatter plot
with st.expander("📈 Scatter Plot: Actual vs Predicted Salary"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Actual vs Predicted Salary")
    st.pyplot(fig)

# Line plot
with st.expander("📊 Line Plot: Actual vs Predicted Salary"):
    fig, ax = plt.subplots()
    ax.plot(y_test.reset_index(drop=True), label="Actual", marker='o', color='blue')
    ax.plot(pd.Series(y_pred), label="Predicted", marker='x', color='red')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Salary")
    ax.set_title("Actual vs Predicted Salary Over Samples")
    ax.legend()
    st.pyplot(fig)