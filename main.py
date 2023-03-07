import pandas as pd
from NeuralNetwork import NeuralNetwork


class NN_interface:
      # Todo: n_features = len(x_columns)
      def __init__(self, model_name, x, y, Ownpred_x=[], Ownpred_y=[],
                   epoch=10, batch_size=20,loaded_model=False,
                   sequence_length=1, train_split=0.7, further_training=True, scale_type="normal", label_feature_name="y", show_column_name="x"):

            self.epoch = epoch
            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self.loaded_model = loaded_model
            self.further_training = further_training
            self.train_split = train_split
            self.scale_type = scale_type
            self.label_feature_name = label_feature_name
            self.show_column_name = show_column_name

            self.NN = NeuralNetwork(model_name=model_name, x=x, y=y,
                 OwnPred_x=Ownpred_x, OwnPred_y=Ownpred_y)

      def run_rnn(self):
            self.NN.normalize_data(features=[], scale_type=self.scale_type, label_feature_name=self.label_feature_name)
            self.NN.split_train_test(train_split=self.train_split, shuffle=False)
            # it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes
            self.NN.convert_to_array()
            self.NN.call_convert_to_timeseries(sequence_length=self.sequence_length, batch_size=self.batch_size)

            self.NN.build_model(nn_type='rnn',loaded_model=self.loaded_model)

            self.NN.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=self.further_training)
            self.NN.predictNN()
            self.NN.denormalize_data(is_preds_normalized=True)
            self.NN.evaluate()

            # self.NN.convert_to_df() # if we do not denormalize the data (so normalization wasn't called) they won't be converted to pd.dataframes
            self.NN.showTrainTest(with_pred=True, column_name=self.show_column_name)
            self.NN.save_model()

      def run_ann(self):
            self.NN.normalize_data(features=[], scale_type=self.scale_type)
            self.NN.split_train_test(train_split=self.train_split, shuffle=True)

            self.NN.build_model(nn_type='ann',loaded_model=self.loaded_model)
            self.NN.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=self.further_training)
            self.NN.predictNN()

            self.NN.denormalize_data(is_preds_normalized=True)

            self.NN.evaluate()
            self.NN.showTrainTest(with_pred=True, column_name=self.show_column_name)
            self.NN.save_model()


# ToDO write Tester class with unittests.

# from GenerateData import genSinwawe
# data = genSinwawe(2, 1340)
# from GenerateData import genUnNormalizedData
# data = genUnNormalizedData(-800, 800,type='square', step=1)
data = pd.read_excel("C:\Egyetem\Diplomamunka\data\TanulokAdatSajat.xlsx")

# data = pd.DataFrame(data)
x_columns = data.columns[:-2]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
if not type(y_columns)==str:
      y_columns = [col for col in y_columns]

# these parameters will come from a UI or config-file.
n_input = 6
batch_size = 6
epoch = 200
train_split = 0.75
nn_type = 'ann'
model_name = "timeseries_seasonal"
label_feature_name = "y"
scale_type = "normal"

show_column_name = "TanOra"

if __name__ == "__main__":
      Runner = NN_interface(model_name,
                            data[x_columns], data[y_columns],
                            epoch=epoch, batch_size=batch_size,
                            sequence_length=n_input, train_split=train_split,
                            label_feature_name=label_feature_name, scale_type=scale_type, show_column_name=show_column_name)
      if nn_type == "ann":
            Runner.run_ann()
      elif nn_type == "rnn":
            Runner.run_rnn()

