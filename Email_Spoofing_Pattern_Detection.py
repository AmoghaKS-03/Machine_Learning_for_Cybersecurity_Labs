# Experiment 8: Email Spoofing Pattern Detection

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Create demonstration email dataset
data = {
    "sender_domain": [
        "google.com", "microsoft.com", "amazon.com",
        "google.com", "paypal.com", "amazon.com",
        "fake-paypal.com", "secure-bank.com",
        "google-login.com", "fake-amazon.net",
        "microsoft.com", "fake-google.net"
    ],

    "reply_domain": [
        "google.com", "microsoft.com", "amazon.com",
        "unknown.net", "fake-paypal.com", "amazon.com",
        "unknown.net", "unknown.net",
        "unknown.net", "unknown.net",
        "microsoft.com", "unknown.net"
    ],

    "spf_pass": [
        1, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 1, 0
    ],

    "dkim_pass": [
        1, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 1, 0
    ],

    # 0 = Legitimate, 1 = Spoofed (demonstration labels)
    "label": [
        0, 0, 0, 1, 1, 0,
        1, 1, 1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

print("EMAIL DATASET")
print(df.head())

# 2. Feature Engineering
df["domain_mismatch"] = (
    df["sender_domain"] != df["reply_domain"]
).astype(int)

features = [
    "domain_mismatch",
    "spf_pass",
    "dkim_pass"
]

X = df[features]
y = df["label"]

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Train Model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("\nAccuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Potentially Spoofed"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 7. Display Potentially Spoofed Emails
df["Prediction"] = model.predict(X)

df["Prediction"] = df["Prediction"].map({
    0: "Legitimate",
    1: "Potentially Spoofed"
})

print("\nEMAIL DETECTION RESULTS")
print(df[[
    "sender_domain",
    "reply_domain",
    "spf_pass",
    "dkim_pass",
    "Prediction"
]].to_string(index=False))