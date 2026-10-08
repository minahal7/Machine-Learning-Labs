import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load diabetes dataset
data = load_diabetes()

X_feature = data.data
Y_Target = data.target

# Training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X_feature,
    Y_Target,
    test_size=0.2,
    random_state=42
)

# K values
k_values = [1, 3, 5, 7, 9]
r2_scores = []

# Train KNN for each k
for k in k_values:

    knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(X_train, Y_train)

    r2 = knn.score(X_test, Y_test)

    r2_scores.append(r2)

# Print results
print("K values:", k_values)
print("R² scores:", r2_scores)

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(k_values, r2_scores)

plt.xlabel("K Values")
plt.ylabel("R² Score")
plt.title("KNN Regression: R² Score vs K Values")

plt.grid(True)
plt.show()