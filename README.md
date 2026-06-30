import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = {
    "Size": [800,1000,1200,1500,1800,2000,2200,2500],
    "Bedrooms": [2,2,3,3,4,4,5,5],
    "Age": [20,15,10,8,5,3,2,1],
    "Price": [45,55,65,80,95,115,130,150]
}

df = pd.DataFrame(data)

X = df[["Size","Bedrooms","Age"]]
Y = df["Price"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(Y_test, lr_pred))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(Y_test, rf_pred))

print("Linear Regression RMSE:", lr_rmse)
print("Random Forest RMSE:", rf_rmse)

plt.scatter(Y_test, lr_pred, label="Linear")
plt.scatter(Y_test, rf_pred, label="RF")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
