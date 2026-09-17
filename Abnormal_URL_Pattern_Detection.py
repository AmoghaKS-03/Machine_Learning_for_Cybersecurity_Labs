# Experiment 6: Abnormal URL Pattern Detection
# Using Feature Engineering and Isolation Forest

import pandas as pd
import re
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# 1. Create URL Dataset

urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.github.com",
    "https://www.wikipedia.org",
    "https://www.amazon.com",
    "https://www.apple.com",
    "https://www.python.org",
    "https://www.youtube.com",
    "https://www.linkedin.com",
    "https://www.facebook.com",

    "http://192.168.1.10/login",
    "http://example.com/login?user=admin&password=1234",
    "http://secure-login-example.com/verify/account",
    "http://free-prize-example.com/winner/claim-now",
    "http://example.com/a/b/c/d/e/f/g/h/i/j/k",
    "http://example.com/login.php?id=123456789&session=987654321",
    "http://example.com/@admin/login",
    "http://192.168.10.20:8080/admin/login",
    "http://example.com/download.php?file=abc&user=test&id=123",
    "http://very-long-domain-name-example.com/account/verify/login/password"
]

df = pd.DataFrame(urls, columns=["URL"])


# 2. Feature Engineering

def extract_features(url):

    parsed = urlparse(url)

    hostname = parsed.hostname if parsed.hostname else ""
    path = parsed.path
    query = parsed.query

    return [
        len(url),                         # URL length
        url.count("."),                   # Number of dots
        url.count("/"),                   # Number of slashes
        url.count("?"),                   # Number of ?
        url.count("="),                   # Number of =
        url.count("-"),                   # Number of hyphens
        url.count("@"),                   # Number of @
        url.count("%"),                   # Number of %
        sum(c.isdigit() for c in url),    # Number of digits
        len(re.findall(r"[^a-zA-Z0-9]", url)),  # Special characters
        int(parsed.scheme == "https"),    # HTTPS
        int(bool(re.match(
            r"^(?:\d{1,3}\.){3}\d{1,3}$",
            hostname
        ))),                              # IP address
        max(0, len(hostname.split(".")) - 2)
            if hostname else 0,            # Subdomains
        len(query),                       # Query length
        len(path),                        # Path length
        len(query.split("&")) if query else 0  # Parameters
    ]


feature_names = [
    "URL_Length",
    "Num_Dots",
    "Num_Slashes",
    "Num_Question",
    "Num_Equal",
    "Num_Hyphen",
    "Num_At",
    "Num_Percent",
    "Num_Digits",
    "Num_Special",
    "Has_HTTPS",
    "Has_IP",
    "Subdomain_Count",
    "Query_Length",
    "Path_Length",
    "Num_Parameters"
]

features = df["URL"].apply(extract_features)

feature_df = pd.DataFrame(
    features.tolist(),
    columns=feature_names
)

print("ENGINEERED FEATURES")
print(feature_df)


# 3. Feature Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(feature_df)


# 4. Isolation Forest

model = IsolationForest(
    contamination=0.20,
    random_state=42
)

model.fit(X_scaled)


# 5. Detect Normal and Abnormal URLs

predictions = model.predict(X_scaled)

result_df = df.copy()

result_df["Status"] = predictions

result_df["Status"] = result_df["Status"].map({
    1: "Normal",
    -1: "Abnormal"
})


# 6. Calculate Anomaly Score

result_df["Anomaly_Score"] = \
    model.decision_function(X_scaled)


# 7. Display Results

print("\nURL ANOMALY DETECTION RESULTS")

print(
    result_df[
        ["URL", "Anomaly_Score", "Status"]
    ].sort_values("Anomaly_Score")
)


# 8. Display Abnormal URLs

print("\nDETECTED ABNORMAL URLs")

abnormal_urls = result_df[
    result_df["Status"] == "Abnormal"
]

print(
    abnormal_urls[
        ["URL", "Anomaly_Score", "Status"]
    ].sort_values("Anomaly_Score")
)


# 9. Visualization

normal_count = (
    result_df["Status"] == "Normal"
).sum()

abnormal_count = (
    result_df["Status"] == "Abnormal"
).sum()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal", "Abnormal"],
    [normal_count, abnormal_count]
)

plt.xlabel("URL Category")
plt.ylabel("Number of URLs")
plt.title("Normal vs Abnormal URL Patterns")

plt.show()