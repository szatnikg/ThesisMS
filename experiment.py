
import keras.layers
import numpy as np
import pandas as pd
# from pandas import DataFrame as df
from matplotlib import pyplot as plt
from NN_recurrent import InputProcessing


from GenerateData import genSinwawe
data = genSinwawe(0.5, 800)

# data = {"x": [i for i in range(500)],
#         "y": [j**2 for j in range(500)]}

data = pd.DataFrame(data)
x_columns = data.columns[:-1]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
y_columns = [col for col in y_columns]

# Note:
# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)


n_features = len(x_columns)
n_input = 10
batch_size = 10
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
from tensorflow import data,keras


""" --------------------------------------------------------"""
# generator = TimeseriesGenerator(processer.x_train, y_deviated, length=n_input, batch_size=batch_size)

# sequence_stride here is the same as time_series_deviation.
my_data = np.array(processer.x_test, dtype=np.float32)
pred_data = keras.utils.timeseries_dataset_from_array(
        data=processer.x_test,
      targets=None,
      sequence_length=n_input,
      sequence_stride=1,
      shuffle=False,
      batch_size=batch_size)
fit_data = keras.utils.timeseries_dataset_from_array(
        data=processer.x_train,
      targets=processer.y_train,
      sequence_length=n_input,
      sequence_stride=1,
      shuffle=False,
      batch_size=batch_size)
# for i,y in fit_data:
#     print("fit_timeseries:" ,i,y)



# d = data.Dataset.from_tensor_slices(x_test)
# here x_test should regained back, because test_generator is not appropiate for model.predict
# but for scores = model.evaluate_generator(test_generator) this is good.

# define model
model = Sequential()
model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
model.add(LSTM(60, activation='relu', return_sequences=False, input_shape=(n_input, n_features)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
# print(processer.x_test.shape)
# a=b=a
model.fit(fit_data, epochs=300, verbose=0, shuffle=True)
# Model was constructed with shape (None, 4, 2) -- (len(x_test), timestamps_in_sequence, features)
print("x_train",processer.x_train)
print("x_test",processer.x_test)
preds = model.predict(pred_data, verbose=0) # processer.x_test.reshape(len(processer.x_test),n_input,n_features)
preds = pd.DataFrame(preds, columns=["preds"])

a,b,c,d, preds = processer.denormalize_data(preds=preds, is_preds_normalized=True)

plt.plot(processer.x_test[:len(preds)], preds, c="r", label="pred")
plt.plot(processer.x_test, processer.y_test, c ="g",label="test")
plt.plot(processer.x_train, processer.y_train, c ="b",label="train")
plt.show()

""" --------------------------------------------------------"""


# define dataset for multivariate timeseriesgeneration
# dataset = hstack((series, series2))
# print(dataset)
# n_features = dataset.shape[1]
