import pandas as pd
from NeuralNetwork import NeuralNetwork
from loader import LoadConfig

class NN_interface(LoadConfig):

      def __init__(self, x, y, Ownpred_x=[], Ownpred_y=[],
                   x_columns=["x"], y_columns=["y"]):
            super(NN_interface, self).__init__()

            self.NN = NeuralNetwork(model_name=self.model_name, x=x, y=y,
                 OwnPred_x=Ownpred_x, OwnPred_y=Ownpred_y, network_structure=self.config_file,
                                    x_columns=x_columns, y_columns=y_columns)

            if self.nn_type == "rnn":
                  self.run_rnn()
            elif self.nn_type == "ann":
                  self.run_ann()
            else: raise ValueError("Define nn_type from: [ann, rnn]")

      def run_rnn(self):
            self.NN.normalize_data(features=[], is_normalize=self.is_normalize, scale_type=self.scale_type, label_feature_name=self.label_feature_name)
            self.NN.split_train_test(train_split=self.train_split, shuffle=False)
            # it produces x_train, x_test, y_train, y_test if shuffle was true, in shuffled format as pd.dataframes
            self.NN.convert_to_array()
            self.NN.call_convert_to_timeseries(sequence_length=self.sequence_length, batch_size=self.batch_size)

            self.NN.build_model(nn_type='rnn', loaded_model=self.loaded_model)

            self.NN.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=self.further_training)
            self.NN.predictNN()
            if self.NN.preprocessed:
                  self.NN.denormalize_data(is_preds_normalized=True)
            else: self.NN.convert_to_df()
            self.NN.evaluate()

            # self.NN.convert_to_df() # if we do not denormalize the data (so normalization wasn't called) they won't be converted to pd.dataframes
            if self.show_plot : self.NN.showTrainTest(with_pred=True, column_name=self.show_column_name)
            self.NN.save_model(model_lib=self.model_lib)

      def run_ann(self):
            self.NN.normalize_data(features=[], is_normalize=self.is_normalize, scale_type=self.scale_type, label_feature_name=self.label_feature_name)
            self.NN.split_train_test(train_split=self.train_split, shuffle=self.shuffle)

            self.NN.build_model(nn_type='ann',loaded_model=self.loaded_model)
            self.NN.train_network(epoch=self.epoch, batch_size=self.batch_size, further_training=self.further_training)
            self.NN.predictNN()
            if self.NN.preprocessed:
                  self.NN.denormalize_data(is_preds_normalized=True)
            else: self.NN.convert_to_df()

            self.NN.evaluate()
            if self.show_plot:
                  self.NN.showValLoss()
                  self.NN.showTrainTest(with_pred=True, column_name=self.show_column_name)
            self.NN.save_model(model_lib=self.model_lib)

      @staticmethod
      def compare_performance( x_test, y_test, preds, output_full_path):
            # run comparison for test_files and individual purposes
            # x_test, y_test should suffer index resetting if they were shuffled!
            # use reset_index(drop=False) on pd.Dataframe object.
            # output file is in csv format

            comparision_df = pd.concat([x_test, y_test, preds], axis=1)
            comparision_df["rel_error"] = abs(y_test - preds) / y_test
            comparision_df["rel_error_percent"] = (abs(y_test - preds) / y_test ) * 100
            comparision_df["abs_error"] = abs(y_test - preds) / (y_test.max() - y_test.min())
            comparision_df["abs_error_percent"] = (abs(y_test - preds) / (y_test.max() - y_test.min()) )*100

            comparision_df.to_csv(output_full_path + ".csv", index=False)
            return comparision_df


# ToDO write Tester class with unittests.

# from GenerateData import genSinwawe
# data = genSinwawe(2, 1340)
from GenerateData import genUnNormalizedData
data = genUnNormalizedData(0, 800,type='square', step=1)
# data = pd.read_excel("C:\Egyetem\Diplomamunka\data\TanulokAdatSajat.xlsx")

# data = pd.DataFrame(data)
x_columns = data.columns[:-1]
x_columns = [col for col in x_columns]
y_columns = data.columns[-1]
if not type(y_columns) == str:
      y_columns = [col for col in y_columns]

# these parameters will come from a UI or config-file.
# n_input = 6
# batch_size = 6
# epoch = 200
# train_split = 0.75

# model_name = "qubic_with_minus"
# label_feature_name = "y"
# scale_type = "normal"
# loaded_model = False
# show_column_name = "x"

if __name__ == "__main__":
      Runner = NN_interface(data[x_columns], data[y_columns], x_columns=x_columns, y_columns=y_columns)
