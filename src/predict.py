import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/churn.csv")

data["churn"] = data["churn"].map({"Yes": 1, "No": 0})
data = pd.get_dummies(data, columns=["contract_length"])

X = data.drop("churn", axis=1)
y = data["churn"]

model = LogisticRegression()
model.fit(X, y)

age = int(input("Enter age: "))
monthly_charges = float(input("Enter monthly charges: "))
support_calls = int(input("Enter number of support calls: "))

print("\nContract type:")
print("1 = month-to-month")
print("2 = one-year")
print("3 = two-year")

contract_choice = int(input("Choose contract type: "))

month_to_month = 0
one_year = 0
two_year = 0

if contract_choice == 1:
    month_to_month = 1
elif contract_choice == 2:
    one_year = 1
elif contract_choice == 3:
    two_year = 1

new_customer = pd.DataFrame({
    "age": [age],
    "monthly_charges": [monthly_charges],
    "support_calls": [support_calls],
    "contract_length_month-to-month": [month_to_month],
    "contract_length_one-year": [one_year],
    "contract_length_two-year": [two_year]
})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nPrediction: Customer is likely to churn.")
else:
    print("\nPrediction: Customer is likely to stay.")