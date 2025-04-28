
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [40000, 45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000],
    'Area': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Age': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Price': [200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000]
}

df = pd.DataFrame(data)

print("---- Simple Linear Regression ----")

X_simple = df[['YearsExperience']]
y_simple = df['Salary']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=0)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_s = model_simple.predict(X_test_s)


print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R²:", r2_score(y_test_s, y_pred_s))


plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', label='Predicted')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

print("\n---- Multiple Linear Regression ----")

X_multi = df[['Area', 'Bedrooms', 'Age']]
y_multi = df['Price']


X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=0)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_m = model_multi.predict(X_test_m)

print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R²:", r2_score(y_test_m, y_pred_m))
