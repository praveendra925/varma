import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/PRAVEENDRA/Downloads/2.csv")  
X = df[['YearsExperience']] 
y = df['Salary']            
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
intercept = model.intercept_
coefficient = model.coef_[0]

print(f"Intercept (b₀): {intercept}")
print(f"Coefficient (b₁): {coefficient}")
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title('Simple Linear Regression: Salary vs YearsExperience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
