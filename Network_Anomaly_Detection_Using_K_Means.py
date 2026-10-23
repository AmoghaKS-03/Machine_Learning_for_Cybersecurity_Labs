import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample network traffic dataset
data = {
    "packet_size": [
        500, 520, 480, 510, 495, 530, 505, 490,
        515, 500, 2000, 2500, 3000, 1800, 4000
    ],
    "connection_duration": [
        10, 12, 9, 11, 10, 13, 10, 9,
        12, 11, 80, 100, 120, 75, 150
    ],
    "connection_count": [
        5, 6, 4, 5, 5, 6, 5, 4,
        6, 5, 40, 50, 60, 35, 80
    ]
}

df = pd.DataFrame(data)

# Select features
X = df[[
    "packet_size",
    "connection_duration",
    "connection_count"
]]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Calculate distance from each cluster center
distances = kmeans.transform(X_scaled)
df["Distance"] = distances.min(axis=1)

# Flag unusually distant events
threshold = df["Distance"].quantile(0.90)
df["Status"] = df["Distance"].apply(
    lambda d: "Potentially Abnormal"
    if d > threshold else "Normal"
)

# Display results
print("NETWORK ANOMALY DETECTION RESULTS")
print(df.to_string(index=False))

# Visualize network events
plt.scatter(
    df["packet_size"],
    df["connection_count"],
    c=df["Cluster"]
)
plt.xlabel("Packet Size")
plt.ylabel("Connection Count")
plt.title("Network Anomaly Detection Using K-Means")
plt.show()