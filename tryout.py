import pandas as pd
from pathlib import Path
import os

proj_folder = Path().absolute() # Path(__file__).parent.resolve()
data_folder = os.path.join(proj_folder,".." , "project_data")

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# # define dataset for multivariate timeseriesgeneration
# # dataset = hstack((series, series2))
# # print(dataset)
# # n_features = dataset.shape[1]

# Note:
# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)
import json
class Source:
    def __init__(self, config_path=None):

        if config_path: self.config_path = config_path
        else: self.config_path = os.path.join(proj_folder,"config.json")

        # JSON keys in following order: Input layer spec, Hidden layer spec, Hyper parameter spec.
        self.KEY_input_spec = "input_layer"
        self.KEY_input_type = "type"
        self.KEY_input_shape_i = "shape_1"
        self.KEY_input_shape_ii = "shape_2"

        self.KEY_hidden_spec = "hidden_layers"
        self.KEY_hidden_type = "type"
        self.KEY_hidden_unit = "unit"
        self.KEY_hidden_return_seq = "return_sequences"
        self.KEY_hidden_activation = "activation"
        self.KEY_hidden_initializer = "initializer"

        self.KEY_hyper_model_name = "model_name"
        self.KEY_hyper_epoch = "epoch"
        self.KEY_hyper_batch_size = "batch_size"
        self.KEY_hyper_loaded_model = "loaded_model"
        self.KEY_hyper_seq_length = "sequence_length"
        self.KEY_hyper_train_split = "train_split"
        self.KEY_hyper_further_train = "further_training"
        self.KEY_hyper_label_feature_name = "label_feature_name_for_normalization"
        self.KEY_hyper_show_column_name = "show_column_name_in_plot"
        self.KEY_hyper_scale_type = "scale_type"

        self.read_config()

    def read_config(self):
        with open(self.config_path, "r") as json_file:
            self.config_file = json.load(json_file)

        self.model_name = self.config_file[self.KEY_hyper_model_name]
        self.epoch = self.config_file[self.KEY_hyper_epoch]
        self.batch_size = self.config_file[self.KEY_hyper_batch_size]
        self.loaded_model = self.config_file[self.KEY_hyper_loaded_model]
        self.sequence_length = self.config_file[self.KEY_hyper_seq_length]
        self.train_split = self.config_file[self.KEY_hyper_train_split]
        self.further_training = self.config_file[self.KEY_hyper_further_train]
        self.label_feature_name = self.config_file[self.KEY_hyper_label_feature_name]
        self.show_column_name = self.config_file[self.KEY_hyper_show_column_name]
        self.scale_type = self.config_file[self.KEY_hyper_scale_type]


    def write_config(self):
        with open("project_data-folder..+self.config_path", "w") as json_output:
            json.dump(self.config_file,json_output)

a = Source("config.json")
a=b=a
if __name__ == "__main__":

    # data = pd.read_excel(
    #     "C:\Egyetem\Diplomamunka\data\TanulokAdatSajat.xlsx")
    # data = GenerateData.genUnNormalizedData(1,300,type="square",step=400)
    # own_pred_data = pandas.read_excel("C:\Egyetem\Diplomamunka\data\TanulokAdatSajat_ownpred.xlsx")

    pred_x = [] # own_pred_data.iloc[::, :-2]
    pred_y = [] #own_pred_data.iloc[::,-2]  #pandas.DataFrame({"y": [j**2 for j in range(460, 560)]})
    import keras
    config_file = {
            "input_layer": { "type": "Dense",
                            "shape_1": "n_features",
                            "shape_2": "None"
                            },

            "hidden_layers": [{"type": "Dense",
                               "unit": 64,
                               "initializer": "random_normal",
                               "activation": "relu",
                               "return_sequences": "false"
                               },
                              {"type": "Dense",
                               "unit": 32,
                               "initializer": "random_normal",
                               "activation": "relu",
                               "return_sequences": "false"
                               },
                              {"type": "Dense",
                               "unit": 1,
                               "initializer": "None",
                               "activation": "None",
                               "return_sequences": "None"
                               }
                              ]
        }

    model = keras.Sequential()
    for key, value in config_file.items():
        if key == "input_layer":
            shape_I = value["shape_1"]
            shape_II = value["shape_2"]
            if shape_I.lower() == "none":
                shape_I = None
            elif shape_I.lower() == "n_features":
                shape_I = 1
            if shape_II.lower() == "none":
                shape_II = None
            elif shape_II.lower() == "n_features":
                shape_II = 1

            if value["type"].lower() == "dense":
                input_layer = keras.Input(shape=(shape_I,))
            elif value["type"].lower() == "lstm":
                input_layer = keras.Input(shape=(None, shape_II))
            else: raise ValueError("This layer type is not available choose from: [Dense, LSTM]")

            model.add(input_layer)
        if key == "hidden_layers":
            for layer in value:
                nn_type = layer["type"]
                unit = layer["unit"]
                activation_func = layer["activation"]
                if activation_func.lower() == "none":
                    activation_func = None

                return_sequences = layer["return_sequences"]
                if return_sequences.lower() == "none":
                    return_sequences = None
                elif return_sequences.lower() == "true":
                    return_sequences = True
                else: return_sequences = False

                if nn_type.upper() == "DENSE":
                    hidden_lays = keras.layers.Dense(unit, activation=activation_func,
                                                     kernel_initializer=keras.initializers.random_normal)
                elif nn_type.upper() == "LSTM":
                    hidden_lays = keras.layers.LSTM(unit, activation=activation_func, return_sequences=return_sequences)
                else:
                    raise ValueError("This layer type is not available choose from: [Dense, LSTM]")
                model.add(hidden_lays)
print(model.summary())

    # This applies to only OwnPred usage.
    # shuffle order is right for the showTrainTest method, but using pandas.concat,
    # the indexes are applied, so x_test,y_test must be resetted.
    # output = pandas.concat([NN.x_test.reset_index(inplace=False, drop=False),
    #                         NN.y_test.reset_index(inplace=False, drop=False), NN.preds], axis= 1)
    #
    # output.to_csv("result.csv")
