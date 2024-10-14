# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('student_data.csv')

# Preprocessing
data_encoded = pd.get_dummies(data, drop_first=True)

# Features: G1, G2, G3
X = data[['G1', 'G2', 'G3']]

# Target variable: number of failures
y = data['failures']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X_train, y_train)


# Make predictions
y_pred = lin_reg.predict(X_test)

# Evaluate the model
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f'RMSE: {rmse:.2f}')
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lin_reg.coef_
})

print(coefficients)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Number of Failures')
plt.ylabel('Predicted Number of Failures')
plt.title('Actual vs. Predicted Failures')
plt.show()


residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

