import pandas as pd

data = pd.read_csv("data/churn.csv")

data["churn"] = data["churn"].map({"Yes": 1, "No": 0})

data = pd.get_dummies(data, columns=["contract_length"])

print(data.head())