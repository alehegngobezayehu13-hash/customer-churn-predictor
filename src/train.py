import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

print("Model trained successfully!")