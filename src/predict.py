import pandas as pd
import pickle

with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

credit_score = int(input("Enter credit score: "))
age = int(input("Enter age: "))
tenure = int(input("Enter tenure: "))
balance = float(input("Enter balance: "))
num_products = int(input("Enter number of products: "))
has_card = int(input("Has credit card? (1=yes, 0=no): "))
is_active = int(input("Is active member? (1=yes, 0=no): "))
salary = float(input("Enter estimated salary: "))
complain = int(input("Number of complaints: "))
satisfaction = int(input("Satisfaction score: "))
points = int(input("Points earned: "))

print("\nGeography:")
print("1 = Germany")
print("2 = Spain")
print("3 = France")

geo_choice = int(input("Choose geography: "))

geo_germany = 0
geo_spain = 0

if geo_choice == 1:
    geo_germany = 1
elif geo_choice == 2:
    geo_spain = 1

gender_input = input("Gender (male/female): ").lower()

gender_male = 0

if gender_input == "male":
    gender_male = 1

print("\nCard Type:")
print("1 = GOLD")
print("2 = PLATINUM")
print("3 = SILVER")
print("4 = DIAMOND")

card_choice = int(input("Choose card type: "))

card_gold = 0
card_platinum = 0
card_silver = 0

if card_choice == 1:
    card_gold = 1
elif card_choice == 2:
    card_platinum = 1
elif card_choice == 3:
    card_silver = 1

new_customer = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary],
    "Complain": [complain],
    "Satisfaction Score": [satisfaction],
    "Point Earned": [points],
    "Geography_Germany": [geo_germany],
    "Geography_Spain": [geo_spain],
    "Gender_Male": [gender_male],
    "Card Type_GOLD": [card_gold],
    "Card Type_PLATINUM": [card_platinum],
    "Card Type_SILVER": [card_silver]
})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nPrediction: Customer is likely to churn.")
else:
    print("\nPrediction: Customer is likely to stay.")