# ============================================
# EXPERIMENT 3
# Random Forest Classification of Cyber Attacks
# Using HIKARI-2021 Dataset
# ============================================


# ============================================
# 1. Import libraries
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 2. Load a representative sample
# ============================================

file_path = "A:\Academic\LABs\III_Sem\MLCS\ALLFLOWMETER_HIKARI2021.csv"

chunks = []

for chunk in pd.read_csv(file_path, chunksize=50000):

    # Take a random sample from each chunk
    sample = chunk.sample(
        n=min(5000, len(chunk)),
        random_state=42
    )

    chunks.append(sample)

# Combine samples
df = pd.concat(chunks, ignore_index=True)

print("Sampled dataset shape:", df.shape)


# ============================================
# 3. Display basic information
# ============================================

print("\nFirst 5 rows:")
print(df.head())

print("\nNumber of columns:", len(df.columns))


# ============================================
# 4. Check target distribution
# ============================================

print("\nAttack / Normal Distribution:")
print(df["Label"].value_counts())


# ============================================
# 5. Remove unnecessary columns
# ============================================

columns_to_drop = [
    "Unnamed: 0.1",
    "Unnamed: 0",
    "uid",
    "origin",
    "origip",
    "resp_h",
    "traffic_category"
]

# Drop only columns that actually exist
columns_to_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

df = df.drop(columns=columns_to_drop)


# ============================================
# 6. Separate features and target
# ============================================

X = df.drop(columns=["Label"])
y = df["Label"]


# ============================================
# 7. Keep numerical features
# ============================================

X = X.select_dtypes(include=[np.number])

print("\nNumber of features used:", X.shape[1])


# ============================================
# 8. Handle infinite and missing values
# ============================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())


# ============================================
# 9. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================
# 10. Create Random Forest Classifier
# ============================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ============================================
# 11. Train the model
# ============================================

rf.fit(X_train, y_train)


# ============================================
# 12. Make predictions
# ============================================

y_pred = rf.predict(X_test)


# ============================================
# 13. Model Evaluation
# ============================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================
# 14. Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# 15. Classification Report
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# ============================================
# 16. Plot Confusion Matrix
# ============================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Normal", "Attack"])
plt.yticks([0, 1], ["Normal", "Attack"])

plt.colorbar()

plt.show()