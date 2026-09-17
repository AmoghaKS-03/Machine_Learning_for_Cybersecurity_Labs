# Experiment 4: Logistic Regression for Malicious URL Detection

import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. Dataset
legitimate_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.apple.com",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.linkedin.com",
    "https://www.netflix.com",
    "https://www.python.org",
    "https://www.ibm.com",
    "https://www.jainuniversity.ac.in",
    "https://www.amazon.in",
    "https://www.flipkart.com",
    "https://www.infosys.com",
    "https://www.wipro.com",
    "https://www.oracle.com",
    "https://www.adobe.com",
    "https://www.cisco.com",
    "https://www.samsung.com",
    "https://www.intel.com"
]

malicious_urls = [
    "http://login-security-verify.com",
    "http://paypal-account-verify.com/login",
    "http://secure-login-update.net",
    "http://banking-security-alert.com",
    "http://free-gift-winner.com/claim",
    "http://account-verification.xyz/login",
    "http://update-password-security.com",
    "http://verify-account-now.net",
    "http://login-confirmation.tk",
    "http://secure-paypal-login.ml",
    "http://bank-alert-verify.xyz",
    "http://win-prize-free.com/claim",
    "http://password-reset-confirm.net",
    "http://security-check-account.com",
    "http://login-verify-user.xyz",
    "http://free-money-winner.com/claim",
    "http://account-update-required.xyz",
    "http://secure-bank-login.net",
    "http://verify-payment-account.com",
    "http://urgent-security-check.xyz"
]

urls = legitimate_urls + malicious_urls
labels = [0] * len(legitimate_urls) + [1] * len(malicious_urls)

df = pd.DataFrame({
    "url": urls,
    "label": labels
})

# 2. Feature Extraction
def extract_features(url):
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_percent": url.count("%"),
        "num_digits": sum(c.isdigit() for c in url),
        "https": int(url.startswith("https")),
        "suspicious_keyword": int(bool(re.search(
            r"(login|verify|secure|account|update|password|"
            r"bank|winner|free|confirm|payment|urgent)",
            url.lower()
        )))
    }

X = pd.DataFrame(df["url"].apply(extract_features).tolist())
y = df["label"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

# 6. Prediction
y_pred = model.predict(X_test_scaled)

# 7. Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Legitimate", "Malicious"]
))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Malicious"]
)

disp.plot()
plt.title("Malicious URL Detection")
plt.show()

# 9. Predict New URL
def predict_url(url):
    features = pd.DataFrame([extract_features(url)])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    print("\nURL:", url)

    if prediction == 0:
        print("Prediction: LEGITIMATE")
    else:
        print("Prediction: MALICIOUS")

    print("Legitimate Probability:",
          round(probability[0] * 100, 2), "%")
    print("Malicious Probability:",
          round(probability[1] * 100, 2), "%")


# Test new URLs
predict_url("https://www.google.com")
predict_url("http://secure-login-verify-account.com")
predict_url("http://free-money-winner.com/claim")
predict_url("https://www.github.com")