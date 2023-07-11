
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('gender.csv')

x = data.iloc[:, 1:4]
y = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

age = int(input('Enter your age: '))
height = int(input('Enter your height in cm: '))
weight = float(input('Enter your weight in kg: '))

user_input = pd.DataFrame({'age': [age], 'height': [height], 'weight': [weight]})
user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled)

accuracy = model.score(x_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

if prediction == 0:
    print("The predicted gender is: Female")
else:
    print("The predicted gender is: Male")


# In[ ]:




