import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = {
    'Fever': [1, 1, 0, 1, 0, 0, 1, 0],
    'Cough': [1, 0, 1, 1, 0, 1, 0, 0],
    'Headache': [1, 1, 0, 1, 0, 0, 1, 0],
    'Flu': [1, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Fever', 'Cough', 'Headache']]
y = df['Flu']

model = GaussianNB()
model.fit(X, y)

test = [[1, 1, 1]]

prediction = model.predict(test)

if prediction[0] == 1:
    print("Patient has Flu")
else:
    print("Patient does not have Flu")
