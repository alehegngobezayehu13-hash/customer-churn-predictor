import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv("data/churn.csv")

data["churn"] = data["churn"].map({"Yes": 1, "No": 0})

data = pd.get_dummies(data, columns=["contract_length"])

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

plt.scatter(y_test, predictions)
plt.xlabel("Actual Churn")
plt.ylabel("Predicted Churn")
plt.title("Actual vs Predicted Churn")

plt.show()

with open("models/churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")