import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model

data = pd.read_csv('assignment1_dataset.csv')
Y = data['house price of unit area']
Y = np.expand_dims(Y, axis=1)
cls = linear_model.LinearRegression()

columns = data.keys()
x_columns = columns[:6]

data['transaction date'] = data['transaction date'].str.slice(0, 4)

for column in x_columns:
    X = data[column]
    X = np.expand_dims(X, axis=1)

    cls.fit(X, Y)
    prediction = cls.predict(X)

    print('Mean Square Error of', column, '=', metrics.mean_squared_error(Y, prediction))


