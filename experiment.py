
import keras.layers
import numpy as np
import pandas as pd
# from pandas import DataFrame as df
from matplotlib import pyplot as plt
from NN_recurrent import InputProcessing


data = {"x": [i for i in range(130)],
        "x2": [k for k in range(130)],
        "y": [j*2 for j in range(130)]}

data = pd.DataFrame(data)
x_columns = data.columns[:-1]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
y_columns = [col for col in y_columns]

# Note:
# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)

# from keras_preprocessing.sequence import TimeseriesGenerator as Gen
# from GenerateData import genSinwawe
# data = genSinwawe(0.5, 800)

n_features = len(x_columns)
n_input = 1
processer = InputProcessing(data[x_columns], data[y_columns], shuffle=False)
processer.normalize_data(features=[])
processer.split_train_test(train_split=0.7)
# it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes

processer.convert_to_array(n_features=len(x_columns))



# data_x, data_y = {"x":data["x"]}, {"y":data["y"]}
y_deviated = InputProcessing.handle_timeseries_deviation(processer.y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator


""" --------------------------------------------------------"""
generator = TimeseriesGenerator(processer.x_train, y_deviated, length=n_input, batch_size=10)
# define model
model = Sequential()
model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
model.add(LSTM(60, activation='relu', return_sequences=False, input_shape=(n_input, n_features)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
# print(processer.x_test.shape)
# a=b=a
model.fit(generator, epochs=300, verbose=0, shuffle=True)
# Model was constructed with shape (None, 4, 2) -- (len(x_test), timestamps_in_sequence, features)

preds = model.predict(processer.x_test.reshape(len(processer.x_test),n_input,n_features), verbose=0)
preds = pd.DataFrame(preds, columns=["preds"])

a,b,c,d, preds = processer.denormalize_data(preds=preds,is_preds_normalized=True)

plt.plot(processer.x_test, preds, c="r", label="pred")
plt.plot(processer.x_test, processer.y_test, c ="g",label="test")
plt.plot(processer.x_train, processer.y_train, c ="b",label="train")
plt.show()

""" --------------------------------------------------------"""


# define dataset for multivariate timeseriesgeneration
# dataset = hstack((series, series2))
# print(dataset)
# n_features = dataset.shape[1]
