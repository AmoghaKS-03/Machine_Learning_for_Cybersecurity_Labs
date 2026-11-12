import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample WSN dataset
data = {
    "timestamp": [
        "10:00", "10:05", "10:10", "10:15",
        "11:00", "11:05", "11:10", "11:15",
        "02:00", "02:05", "02:10", "02:15"
    ],
    "source_ip": [
        "192.168.1.1", "192.168.1.2",
        "192.168.1.3", "192.168.1.4",
        "192.168.1.1", "192.168.1.2",
        "192.168.1.3", "192.168.1.4",
        "10.0.0.5", "10.0.0.6",
        "10.0.0.7", "10.0.0.8"
    ],
    "event_type": [
        "Normal", "Normal", "Data", "Data",
        "Normal", "Data", "Normal", "Data",
        "Flooding", "Unknown", "Flooding", "Unknown"
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Extract time features
df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)

# Encode categorical features
ip_encoder = LabelEncoder()
event_encoder = LabelEncoder()

df["ip_code"] = ip_encoder.fit_transform(df["source_ip"])
df["event_code"] = event_encoder.fit_transform(df["event_type"])

# Select features
X = df[["hour", "ip_code", "event_code"]]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")

# Test a new event
new_event = pd.DataFrame({
    "hour": [2],
    "ip_code": [ip_encoder.transform(["10.0.0.5"])[0]],
    "event_code": [event_encoder.transform(["Flooding"])[0]]
})

result = model.predict(new_event)

print("Event Status:",
      "Malicious" if result[0] == 1 else "Normal")