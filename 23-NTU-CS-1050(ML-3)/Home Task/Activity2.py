import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("23-NTU-CS-1050(ML-3)/Car Price Prediction.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "horsepower",
    "enginesize",
    "curbweight",
    "citympg"
]

X = df[features].copy()
y = df["price"].copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

data = pd.concat([X, y], axis=1)
data.dropna(inplace=True)

X = data[features]
y = data["price"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_linear)
)

linear_r2 = r2_score(y_test, y_pred_linear)


print("\n================ LINEAR REGRESSION ================")
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 4))


# ============================================================
# 6. POLYNOMIAL REGRESSION
# ============================================================

results = []

results.append({
    "Model": "Linear Regression",
    "RMSE": linear_rmse,
    "R2 Score": linear_r2
})


for degree in [2, 3, 4]:

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": f"Polynomial Degree {degree}",
        "RMSE": rmse,
        "R2 Score": r2
    })


# ============================================================
# 7. COMPARE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n================ MODEL COMPARISON ================")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 8. PLOT FITTED CURVES
# ============================================================

# We use horsepower for the X-axis
# Other features are kept at their mean values.

horsepower_range = np.linspace(
    X["horsepower"].min(),
    X["horsepower"].max(),
    300
)


plot_data = pd.DataFrame({
    "horsepower": horsepower_range,
    "enginesize": X["enginesize"].mean(),
    "curbweight": X["curbweight"].mean(),
    "citympg": X["citympg"].mean()
})


# ============================================================
# Linear Regression Curve
# ============================================================

linear_curve = linear_model.predict(plot_data)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["horsepower"],
    y_test,
    alpha=0.6,
    label="Actual Test Data"
)

plt.plot(
    horsepower_range,
    linear_curve,
    label="Linear Regression"
)


# ============================================================
# Polynomial Curves
# ============================================================

for degree in [2, 3, 4]:

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    plot_poly = poly.transform(plot_data)

    prediction = model.predict(plot_poly)

    plt.plot(
        horsepower_range,
        prediction,
        label=f"Polynomial Degree {degree}"
    )


# ============================================================
# FINAL GRAPH
# ============================================================

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()

plt.show()