import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv("Salary Data.csv")
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)

# Feature and target separation
X = data.drop('Salary', axis=1)
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and column names
joblib.dump(model, "salary_prediction_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Save test data for visualization in app
joblib.dump((X_test, y_test, model.predict(X_test)), "test_predictions.pkl")

# Optional: Print model performance
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("R² Score:", r2_score(y_test, model.predict(X_test)))