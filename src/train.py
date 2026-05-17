import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/churn.csv")

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data = pd.get_dummies(
    data,
    columns=["Geography", "Gender", "Card Type"],
    drop_first=True
)

X = data.drop("Exited", axis=1)

y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

with open("models/churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")