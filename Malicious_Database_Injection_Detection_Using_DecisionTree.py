import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample SQL query dataset
data = {
    "query": [
        "SELECT name FROM users",
        "SELECT * FROM products",
        "UPDATE users SET age=20",
        "SELECT id FROM students",
        "DELETE FROM cart WHERE id=5",
        "SELECT * FROM orders",
        "SELECT * FROM users WHERE id=1 OR 1=1",
        "SELECT * FROM users UNION SELECT password FROM admin",
        "SELECT * FROM users WHERE name='admin'--",
        "SELECT * FROM users; DROP TABLE users",
        "SELECT * FROM accounts WHERE id=2",
        "SELECT * FROM users WHERE id=' OR '1'='1"
    ],
    "label": [0,0,0,0,0,0,1,1,1,1,0,1]
}

df = pd.DataFrame(data)

# Extract suspicious features
def extract_features(query):
    return [
        len(query),
        len(re.findall(r"(--|#|;)", query)),
        len(re.findall(r"\b(UNION|DROP|INSERT|DELETE|UPDATE)\b",
                       query, re.I)),
        int(bool(re.search(r"\bOR\s+1\s*=\s*1", query, re.I)))
    ]

X = pd.DataFrame(
    df["query"].apply(extract_features).tolist(),
    columns=["length", "special_chars", "sql_keywords", "tautology"]
)

y = df["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")

# Test a new query
new_query = "SELECT * FROM users WHERE id=1 OR 1=1"
features = pd.DataFrame(
    [extract_features(new_query)],
    columns=X.columns
)

result = model.predict(features)

print("Query:", new_query)
print("Result:", "Malicious" if result[0] == 1 else "Normal")