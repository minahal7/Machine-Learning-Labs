import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("23-NTU-CS-1050(ML-3)/Telco Customer Churn.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)


# ============================================================
# 2. PREPROCESSING
# ============================================================

# TotalCharges contains some blank values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove rows with missing values
df.dropna(inplace=True)

# Select required features
features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService"
]

X = df[features].copy()
y = df["Churn"]


# Encode categorical variables
# Contract and InternetService are categorical

X = pd.get_dummies(
    X,
    columns=["Contract", "InternetService"],
    drop_first=True
)

# Encode target variable
# No = 0
# Yes = 1
y = y.map({"No": 0, "Yes": 1})


print("\nPreprocessed features:")
print(X.head())


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Probability of churn
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 7. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

print("\nROC-AUC:", round(auc, 4))