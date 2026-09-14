# ============================================================
# LINEAR REGRESSION - SUPERVISED LEARNING PROJECT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. Create a synthetic regression dataset
# ------------------------------------------------------------

X, y = make_regression(
    n_samples=200,
    n_features=2,
    noise=15,
    random_state=42
)

print("Regression dataset created successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ------------------------------------------------------------
# 2. Split dataset into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 3. Create and train Linear Regression model
# ------------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 4. Make predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 5. Evaluate the model
# ------------------------------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== LINEAR REGRESSION RESULTS =====")
print("Mean Squared Error (MSE):", round(mse, 4))
print("R-squared (R2):", round(r2, 4))


# ------------------------------------------------------------
# 6. Display actual vs predicted values
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))


# ------------------------------------------------------------
# 7. Visualize actual vs predicted values
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

results.to_csv("linear_regression_results.csv", index=False)

print("\nLinear Regression completed successfully!")
