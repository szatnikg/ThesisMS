
import keras.layers
import numpy as np
import pandas as pd
# from pandas import DataFrame as df
from matplotlib import pyplot as plt
from NN_recurrent import RecurrentNN, InputProcessing
from tensorflow import keras

class recurrent_if():
      # Todo: n_features = len(x_columns)
      def __init__(self, model_name, x, y, Ownpred_x=[], Ownpred_y=[],  epoch=10, batch_size=20, sequence_length=1):
            self.epoch = epoch
            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self.rec_NN = RecurrentNN(model_name=model_name, x=x, y=y,
                 OwnPred_x=Ownpred_x, OwnPred_y=Ownpred_y)

      def run(self):
            self.rec_NN.normalize_data(features=[])
            self.rec_NN.split_train_test(train_split=0.75, shuffle=False)
            # it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes
            self.rec_NN.convert_to_array()
            self.rec_NN.call_convert_to_timeseries(sequence_length=self.sequence_length, batch_size=self.batch_size)

            self.rec_NN.build_model()

            self.rec_NN.train_network(epoch=self.epoch, batch_size=self.batch_size)
            self.rec_NN.predictNN()
            self.rec_NN.evaluate()

            self.rec_NN.convert_to_df()
            self.rec_NN.showTrainTest(with_pred=True)
            self.rec_NN.save_model()
            # plt.plot(processer.x_test[:len(preds)]["x"], preds, c="r", label="pred")

from GenerateData import genSinwawe
data = genSinwawe(5, 1300)
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


# n_features = len(x_columns)
n_input = 15
batch_size = 10

if __name__ == "__main__":
      Runner = recurrent_if("timeseries_seasonal",
                            data[x_columns], data[y_columns],
                            epoch=100, batch_size=batch_size,
                            sequence_length=n_input)
      Runner.run()

# processer = InputProcessing(data[x_columns], data[y_columns])
# processer.normalize_data(features=[])
# processer.split_train_test(train_split=0.4)
# # it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes
# processer.convert_to_array()

# data_x, data_y = {"x":data["x"]}, {"y":data["y"]}
# y_deviated = InputProcessing.handle_timeseries_deviation(processer.y_train)

""" --------------------------------------------------------"""

# # sequence_stride here is the same as time_series_deviation.
# # my_data = np.array(processer.x_test, dtype=np.float32)
# pred_data = keras.utils.timeseries_dataset_from_array(
#         data=processer.x_test,
#       targets=None,
#       sequence_length=n_input,
#       sequence_stride=1,
#       shuffle=False,
#       batch_size=batch_size)
# fit_data = keras.utils.timeseries_dataset_from_array(
#         data=processer.x_train,
#       targets=processer.y_train,
#       sequence_length=n_input,
#       sequence_stride=1,
#       shuffle=False,
#       batch_size=batch_size)
# # for i,y in fit_data:
# #     print("fit_timeseries:" ,i,y)
#
#
#
# # d = data.Dataset.from_tensor_slices(x_test)
# # here x_test should regained back, because test_generator is not appropiate for model.predict
# # but for scores = model.evaluate_generator(test_generator) this is good.
#
# # define model
# model = keras.Sequential()
# model.add(keras.layers.LSTM(30, activation='relu', return_sequences=True, input_shape=(None, n_features)))
#
# model.add(keras.layers.LSTM(30, activation='relu', return_sequences=False))
#
# model.add(keras.layers.Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# # print(processer.x_test.shape)
# # a=b=a
# # model = keras.models.load_model("timeseries_seasonal.h5")
# model.fit(fit_data, epochs=50, verbose=0, shuffle=True)
#
# preds = model.predict(pred_data, verbose=0) # processer.x_test.reshape(len(processer.x_test),n_input,n_features)
# # preds = np.reshape(preds, (len(preds), n_features))
#
# preds = pd.DataFrame(preds, columns=["preds"])
# model.save("timeseries_seasonal.h5")
# a,b,c,d, preds = processer.denormalize_data(preds=preds, is_preds_normalized=True)
#
# plt.plot(processer.x_test[:len(preds)]["x"], preds, c="r", label="pred")
# plt.plot(processer.x_test["x"], processer.y_test, c ="g",label="test")
# plt.plot(processer.x_train["x"], processer.y_train, c ="b",label="train")
# plt.show()
#
# """ --------------------------------------------------------"""
#
#
# # define dataset for multivariate timeseriesgeneration
# # dataset = hstack((series, series2))
# # print(dataset)
# # n_features = dataset.shape[1]
