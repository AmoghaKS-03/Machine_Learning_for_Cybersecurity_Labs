import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Sample email dataset
data = {
    "email": [
        "Win a free lottery prize now",
        "Claim your free gift today",
        "Congratulations you won cash",
        "Click here to get a free reward",
        "Earn money quickly from home",
        "Limited offer buy now",
        "You have won a free vacation",
        "Get cheap loans immediately",
        "Free entry in lucky draw",
        "Urgent claim your prize",
        "Meeting scheduled for tomorrow",
        "Please find the project report",
        "Your assignment is due tomorrow",
        "Let's discuss the project",
        "Please attend the team meeting",
        "The class starts at ten",
        "Your order has been delivered",
        "Can we meet this evening",
        "Please check your email",
        "The report is ready for review"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "not spam", "not spam", "not spam",
        "not spam", "not spam", "not spam",
        "not spam", "not spam", "not spam", "not spam"
    ]
}

df = pd.DataFrame(data)

# Convert email text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test a new email
new_email = ["Congratulations! You won a free prize"]
new_email_features = vectorizer.transform(new_email)

prediction = model.predict(new_email_features)

print("\nNew Email:", new_email[0])
print("Prediction:", prediction[0])