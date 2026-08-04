import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample Dataset
data = {
    'FloorArea': [800, 1000, 1200, 1500, 1800],
    'Rooms': [2, 3, 3, 4, 5],
    'Location': [1, 2, 2, 3, 3],
    'Rent': [15000, 20000, 25000, 30000, 35000]
}

df = pd.DataFrame(data)

X = df[['FloorArea', 'Rooms', 'Location']]
y = df['Rent']

model = LinearRegression()
model.fit(X, y)
area = 1400
rooms = 4
location = 3

prediction = model.predict([[area, rooms, location]])

print("Predicted Apartment Rent =", prediction[0])
