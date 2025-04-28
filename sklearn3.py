import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/PRAVEENDRA/Downloads/2.csv")  
X = df[['YearsExperience']] 
y = df['Salary']             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_[0])
