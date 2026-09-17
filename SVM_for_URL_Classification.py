# Experiment 5: SVM for URL Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset
df = pd.read_csv("A:\Academic\LABs\III_Sem\dataset.csv")

print("Dataset loaded successfully!")
print(df.head())

# 2. Check Dataset
print("\nDataset Shape:", df.shape)
print("\nMissing Values:", df.isnull().sum().sum())
print("\nClass Distribution:")
print(df["Result"].value_counts())

# 3. Prepare Features and Target
X = df.drop(columns=["Result", "index"])
y = df["Result"]

# 4. Convert Labels
# -1 = Malicious → 1
#  1 = Legitimate → 0

y = y.map({
    -1: 1,
     1: 0
})

print("\n0 = Legitimate")
print("1 = Malicious")

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# 6. Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Create SVM Model
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

# 8. Train Model
svm_model.fit(X_train_scaled, y_train)

print("\nSVM model trained successfully!")

# 9. Prediction
y_pred = svm_model.predict(X_test_scaled)

# 10. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n===== SVM RESULTS =====")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

# 11. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Malicious"],
        zero_division=0
    )
)

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 13. Display Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("SVM - URL Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Legitimate", "Malicious"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Malicious"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# 14. Test 5 Samples
sample_features = X_test.iloc[:5]
sample_actual = y_test.iloc[:5]

sample_scaled = scaler.transform(sample_features)
sample_predictions = svm_model.predict(sample_scaled)

print("\n===== SAMPLE CLASSIFICATIONS =====")

for i in range(5):

    actual = (
        "Malicious"
        if sample_actual.iloc[i] == 1
        else "Legitimate"
    )

    predicted = (
        "Malicious"
        if sample_predictions[i] == 1
        else "Legitimate"
    )

    print(f"\nSample {i + 1}")
    print("Actual Class    :", actual)
    print("Predicted Class :", predicted)

# 15. Observation
print("\n===== OBSERVATION =====")
print(
    "SVM successfully classified URLs into "
    "Legitimate and Malicious categories."
)