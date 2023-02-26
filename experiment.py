import random

import keras.layers
import numpy as np
from math import sqrt

import pandas as pd
from pandas import DataFrame as df
from matplotlib import pyplot as plt

import DataProcessing
from NN_recurrent import InputProcessing

x_columns = []
y_columns = []

data = {"x": [i for i in range(1000)],
        "y": [j**3 for j in range(1000)]}

data = pd.DataFrame(data)
x_columns = data.columns[:-1]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
y_columns = [col for col in y_columns]

split = 760

# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)

from keras_preprocessing.sequence import TimeseriesGenerator as Gen
from GenerateData import genSinwawe
# data = genSinwawe(0.5, 800)

n_features = 1
n_input = 1

processer = InputProcessing(data[x_columns], data[y_columns])
data_x, data_y, Ownpred_x, Ownpred_y = processer.normalize_data(features=[])

arr = np.array([1,2,3,4,5,6,7])
yhat = arr.reshape(len(arr), 1)
yhat = pd.DataFrame(yhat, columns=["preds"])

# data_x, data_y = {"x":data["x"]}, {"y":data["y"]}
x = data_x[x_columns].to_numpy()
x = x.reshape(len(x), n_features)
y = data_y[y_columns].to_numpy()
y = y.reshape(len(y), n_features)

x_new = pd.DataFrame(x, columns=x_columns)
y_new = pd.DataFrame(y, columns=y_columns)


x, y, Ownpred_x, Ownpred_y, yhat = processer.denormalize_data(x=x_new, y =y_new, OwnPred_x=[], OwnPred_y=[], preds=yhat, is_preds_normalized=True)
print("after:",x[:5])
print("after",yhat[:5])

a=b=a
x_train = data_x[x_columns].to_numpy()[:split] #dense
x_train = x_train.reshape(len(x_train), n_features) # recurrent NN
y_train = data_y[y_columns].to_numpy()[:split]
y_train = y_train.reshape(len(y_train), n_features)

y_deviated = InputProcessing.handle_timeseries_deviation(y_train)

print(x[:5])
print(y_deviated[:5])


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

# define dataset for multivariate timeseriesgeneration
# dataset = hstack((series, series2))
# print(dataset)
# n_features = dataset.shape[1]



# define generator
generator = TimeseriesGenerator(x_train, y_deviated, length=n_input, batch_size=20)
for i in range(len(generator)):
    j, k = generator[i]
    print('%s => %s' % (j, k))
    if i > 4:
        break


# define model
model = Sequential()
model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
model.add(LSTM(60, activation='relu', return_sequences=False, input_shape=(n_input, n_features)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(generator, epochs=300, verbose=0, shuffle=True)
# Model was constructed with shape (None, 4, 2) -- (len(x_test), timestamps_in_sequence, features)

# make one step prediction out of sample
# x_input = array( [[6],[7]])
# x_input2 = x_input.reshape((len(x_input), n_input, n_features))

yhat = model.predict(x[split:], verbose=0)
yhat = pd.DataFrame(yhat, columns=["preds"])

x = df(x, columns=[x_columns])
y = df(y, columns=[y_columns])
x, y, Ownpred_x, Ownpred_y, yhat = processer.denormalize_data(x=x, y =y, OwnPred_x=[], OwnPred_y=[], preds=yhat, is_preds_normalized=True)
print(x[:5])
plt.plot(x[split:], yhat, c="r", label="pred")
plt.plot(x, y, c ="g",label="train + test")
plt.show()