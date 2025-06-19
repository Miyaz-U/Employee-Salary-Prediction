import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import streamlit as st

# Load the dataset
data = pd.read_csv("Salary Data.csv")

# Preview the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (if few)
data.dropna(inplace=True)

# Convert categorical to numerical (if applicable)
data = pd.get_dummies(data, drop_first=True)

X = data[['Years of Experience']]  # Independent variable(s)
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
plt.scatter(X_test, y_test, color='blue', label='Actual Salary')

# Line plot: model predictions
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Salary')

plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# You can do the same for training data
plt.scatter(X_train, y_train, color='green', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Model')
plt.title("Training Set: Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Save the trained model to a file
joblib.dump(model, 'salary_prediction_model.pkl')

# Load the saved model
loaded_model = joblib.load('salary_prediction_model.pkl')

# Create a DataFrame with the correct column name
X_new = pd.DataFrame([[5]], columns=['Years of Experience'])
predicted_salary = model.predict(X_new)

print(f"Predicted Salary for 5 years experience: {predicted_salary[0]:.2f}")

# Load model
model = joblib.load('salary_prediction_model.pkl')

st.title("Salary Predictor")
experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"Estimated Salary: ₹{prediction[0]:,.2f}")