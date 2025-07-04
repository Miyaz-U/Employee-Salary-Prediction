import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import streamlit as st

st.set_page_config(page_title="Salary Predictor", layout="centered")

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

# Load the dataset
data = pd.read_csv("Salary Data.csv")

# Preview the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (if few)
data.dropna(inplace=True)

# Convert categorical to numerical (if applicable)
data = pd.get_dummies(data, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)

X = data.drop('Salary', axis=1)  # Independent variable(s)
y = data['Salary']             # Dependent/target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
b = model.intercept_

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Learned Linear Equation: Salary = {m:.2f} * YearsExperience + {b:.2f}")

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Scatter plot: actual data
plt.scatter(X_test['Years of Experience'], y_test, color='blue', label='Actual Salary')

# Line plot: model predictions
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Salary')

plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# You can do the same for training data
plt.scatter(X_train['Years of Experience'], y_train, color='green', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Model')
plt.title("Training Set: Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Save the trained model to a file
joblib.dump(model, 'salary_prediction_model.pkl')
# Save the columns used for training. This is crucial for prediction later.
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# Load the saved model
#loaded_model = joblib.load('salary_prediction_model.pkl')

# Create a DataFrame with the correct column name
#X_new = pd.DataFrame([[5]], columns=['Years of Experience'])
#predicted_salary = model.predict(X_new)

#print(f"Predicted Salary for 5 years experience: {predicted_salary[0]:.2f}")

# Load model
#model = joblib.load('salary_prediction_model.pkl')

#st.title("Salary Predictor")
#experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)

#if st.button("Predict Salary", key="predict_button"):
    #prediction = model.predict([[experience]])
    #st.success(f"Estimated Salary: ₹{prediction[0]:,.2f}")

# --- Streamlit App ---
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Hide Streamlit default UI
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>💼 Employee Salary Predictor</h1>
    <p style='text-align: center; color: gray;'>Estimate salaries using AI based on employee profile</p>
    <hr style='border: 1px solid #eee;'>
""", unsafe_allow_html=True)

# Load model & columns
model = joblib.load('salary_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')
original_data = pd.read_csv("Salary Data.csv")
original_data.dropna(inplace=True)

# Input Section
st.header("🧾 Enter Employee Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=1, max_value=65, value=1)
    gender = st.radio("👤 Gender", ["Male", "Female"])
    education_level = st.radio("🎓 Education Level", ["Bachelor's", "Master's", "PhD"])

with col2:
    job_title = st.selectbox("💼 Job Title", original_data['Job Title'].unique())
    experience = st.number_input("📈 Years of Experience", min_value=0.0, step=0.1)

# Prepare data for prediction
new_data_dict = {col: 0 for col in model_columns}
new_data_dict['Age'] = age
new_data_dict['Years of Experience'] = experience

if f'Gender_{gender}' in new_data_dict:
    new_data_dict[f'Gender_{gender}'] = 1
if f'Education Level_{education_level}' in new_data_dict:
    new_data_dict[f'Education Level_{education_level}'] = 1
if f'Job Title_{job_title}' in new_data_dict:
    new_data_dict[f'Job Title_{job_title}'] = 1

X_new = pd.DataFrame([new_data_dict])[model_columns]

# Predict button
if st.button("🔍 Predict Salary"):
    prediction = model.predict(X_new)[0]
    st.success(f"💰 Estimated Salary: ₹{prediction:,.2f}", icon="📢")
    st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)

# Display Model Performance
st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
st.write(f"**Mean Squared Error (MSE):** ${mse:,.2f}")
st.write(f"**R-squared (R²):** ${r2:.4f}")

st.subheader("🧠 Feature Importance")
coef_df = pd.DataFrame({'Feature': model_columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
st.bar_chart(coef_df.set_index('Feature'))

# --- Visualization ---
#st.subheader("📉 Salary vs Experience")

# Scatter + Regression Line
#fig, ax = plt.subplots()
#ax.scatter(data['Years of Experience'], data['Salary'], color='blue', label='Actual Data')
#ax.plot(data['Years of Experience'], model.predict(X), color='red', label='Regression Line')
#ax.set_xlabel("Years of Experience")
#ax.set_ylabel("Salary")
#ax.set_title("Salary vs Years of Experience")
#ax.legend()
#st.pyplot(fig) 