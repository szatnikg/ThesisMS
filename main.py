
from NeuralNetwork import NeuralNetwork


class RecurrentIf:
      # Todo: n_features = len(x_columns)
      def __init__(self, model_name, x, y, Ownpred_x=[], Ownpred_y=[],  epoch=10, batch_size=20, sequence_length=1):
            self.epoch = epoch
            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self.rec_NN = NeuralNetwork(model_name=model_name, x=x, y=y,
                 OwnPred_x=Ownpred_x, OwnPred_y=Ownpred_y)

      def run(self):
            self.rec_NN.normalize_data(features=[])
            self.rec_NN.split_train_test(train_split=0.75, shuffle=False)
            # it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes
            self.rec_NN.convert_to_array()
            self.rec_NN.call_convert_to_timeseries(sequence_length=self.sequence_length, batch_size=self.batch_size)

            self.rec_NN.build_model(nn_type='rnn',loaded_model=False)

            self.rec_NN.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=True)
            self.rec_NN.predictNN()
            self.rec_NN.denormalize_data(is_preds_normalized=True)
            self.rec_NN.evaluate()

            # self.rec_NN.convert_to_df() # if we do not denormalize the data (so normalization wasn't called) they won't be converted to pd.dataframes
            self.rec_NN.showTrainTest(with_pred=True)
            self.rec_NN.save_model()
            # plt.plot(processer.x_test[:len(preds)]["x"], preds, c="r", label="pred")

class DenseIf:
      def __init__(self, model_name, x, y, Ownpred_x=[], Ownpred_y=[],  epoch=10, batch_size=20):
            self.epoch = epoch
            self.batch_size = batch_size
            self.regressor = NeuralNetwork(model_name=model_name, x=x, y=y,
                 OwnPred_x=Ownpred_x, OwnPred_y=Ownpred_y)

      def run(self):
            self.regressor.normalize_data(features=[])
            self.regressor.split_train_test(train_split=0.75, shuffle=True)

            self.regressor.build_model(nn_type='ann',loaded_model=False)
            self.regressor.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=True)
            self.regressor.predictNN()

            self.regressor.denormalize_data(is_preds_normalized=True)

            self.regressor.evaluate()
            self.regressor.showTrainTest(with_pred=True)
            self.regressor.save_model()


# ToDO write Tester class with unittests.

# from GenerateData import genSinwawe
# data = genSinwawe(2, 1340)
from GenerateData import genUnNormalizedData
data = genUnNormalizedData(-800, 800,type='square', step=1)

# data = pd.DataFrame(data)
x_columns = data.columns[:-1]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
y_columns = [col for col in y_columns]

# these parameters will come from a UI or config-file.
n_input = 6
batch_size = 10
epoch = 200
nn = 'ann'
model_name = "cubic_with_minus"


if __name__ == "__main__":
      if nn == 'rnn':
            Runner = RecurrentIf("timeseries_seasonal",
                                  data[x_columns], data[y_columns],
                                  epoch=epoch, batch_size=batch_size,
                                  sequence_length=n_input)
      else:
            Runner = DenseIf(model_name,
                                  data[x_columns], data[y_columns],
                                  epoch=epoch, batch_size=batch_size)

      Runner.run()

