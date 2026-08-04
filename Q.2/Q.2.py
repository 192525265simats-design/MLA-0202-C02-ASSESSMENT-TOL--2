import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample Dataset
data = {
    'Free': [5, 0, 4, 1, 6, 0],
    'Offer': [4, 1, 5, 0, 4, 1],
    'Money': [5, 0, 4, 1, 5, 0],
    'Spam': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Free', 'Offer', 'Money']]
y = df['Spam']

model = LogisticRegression()
model.fit(X, y)

# Test Email
test = [[4, 3, 5]]

result = model.predict(test)

if result[0] == 1:
    print("Spam Email")
else:
    print("Non-Spam Email")
