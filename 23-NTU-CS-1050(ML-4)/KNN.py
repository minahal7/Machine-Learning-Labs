import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load wine dataset
wine_data = load_wine()

X_feature = wine_data.data
Y_Target = wine_data.target

# Training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X_feature,
    Y_Target,
    test_size=0.2,
    random_state=42
)

# K values
k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

# Train KNN for each k
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, Y_train)

    accuracy = knn.score(X_test, Y_test)

    accuracy_scores.append(accuracy)

# Print results
print("K values:", k_values)
print("Accuracy scores:", accuracy_scores)

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(k_values, accuracy_scores)

plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K Values")

plt.grid(True)
plt.show()