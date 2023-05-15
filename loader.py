import pandas as pd
from pathlib import Path
import os
from keras import layers, Input, initializers, Sequential
# Note:
# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)
# This applies to only OwnPred usage.
# shuffle order is right for the showTrainTest method, but using pandas.concat,
# the indexes are applied, so x_test,y_test must be resetted.
# output = pd.concat([NN.x_test.reset_index(inplace=False, drop=False),
#                         NN.y_test.reset_index(inplace=False, drop=False), NN.preds], axis= 1)
# output.to_csv("result.csv")
import json
class Source:
    def __init__(self, config_path=None):
        # initialize folders
        proj_folder = Path().absolute()  # alternative: Path(__file__).parent.resolve()

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
        self.KEY_hyper_shuffle = "shuffle"
        self.KEY_hyper_is_normalize = "want_to_normalize"
        self.KEY_hyper_nn_type = "high_level_nn_type"
        self.KEY_hyper_model_lib = "model_lib"
        self.KEY_hyper_show_chart = "show_plot"
        self.KEY_hyper_learning_r = "learning_rate"
        self.KEY_classification = "classification"

class LoadConfig(Source):
    def __init__(self,config_path=None):
        super(LoadConfig, self).__init__(config_path=config_path)

        self.read_config()

    def read_config(self):
        with open(self.config_path, "r") as json_file:
            self.config_file = json.load(json_file)

        self.model_name = self.config_file[self.KEY_hyper_model_name]
        self.show_plot = self.config_file[self.KEY_hyper_show_chart]
        self.epoch = self.config_file[self.KEY_hyper_epoch]
        self.batch_size = self.config_file[self.KEY_hyper_batch_size]
        self.loaded_model = self.config_file[self.KEY_hyper_loaded_model]
        self.sequence_length = self.config_file[self.KEY_hyper_seq_length]
        self.train_split = self.config_file[self.KEY_hyper_train_split]
        self.further_training = self.config_file[self.KEY_hyper_further_train]
        self.learning_rate = self.config_file[self.KEY_hyper_learning_r]
        self.classification = self.config_file[self.KEY_classification]
        self.label_feature_name = self.config_file[self.KEY_hyper_label_feature_name]
        self.show_column_name = self.config_file[self.KEY_hyper_show_column_name]
        self.scale_type = self.config_file[self.KEY_hyper_scale_type]
        self.shuffle = self.config_file[self.KEY_hyper_shuffle]
        self.is_normalize = self.config_file[self.KEY_hyper_is_normalize]
        self.nn_type = self.config_file[self.KEY_hyper_nn_type]
        self.model_lib = self.config_file[self.KEY_hyper_model_lib]

class Layers(Source):
    def __init__(self, layer_obj={}):
        super(Layers, self).__init__()
        # Key names for NN.build_model() method
        # self.KEY_input_spec = "input_layer"
        # self.KEY_input_type = "type"
        # self.KEY_input_shape_i = "shape_1"
        # self.KEY_input_shape_ii = "shape_2"
        #
        # self.KEY_hidden_spec = "hidden_layers"
        # self.KEY_hidden_type = "type"
        # self.KEY_hidden_unit = "unit"
        # self.KEY_hidden_return_seq = "return_sequences"
        # self.KEY_hidden_activation = "activation"
        # self.KEY_hidden_initializer = "initializer"
        self.layer_obj = layer_obj

    def create_input_layer(self, n_features):
        # Input layer decisions are not thought through
        value = self.layer_obj[self.KEY_input_spec]

        shape_I = value["shape_1"]
        shape_II = value["shape_2"]
        if shape_I.lower() == "none":
            shape_I = None
        elif shape_I.lower() == "n_features":
            shape_I = n_features
        if shape_II.lower() == "none":
            shape_II = None
        elif shape_II.lower() == "n_features":
            shape_II = n_features

        if value[self.KEY_input_type].lower() == "dense":
            input_layer = Input(shape=(shape_I,))
        elif value[self.KEY_input_type].lower() == "lstm":
            input_layer = Input(shape=(None, shape_II))
        else: raise ValueError("This layer type is not available choose from: [Dense, LSTM]")

        return input_layer

    def generate_hidden_layer(self):

        value = self.layer_obj[self.KEY_hidden_spec]

        for layer in value:
            nn_type = layer[self.KEY_hidden_type]
            unit = layer[self.KEY_hidden_unit]
            activation_func = layer[self.KEY_hidden_activation]
            if activation_func.lower() == "none":
                activation_func = None

            return_sequences = layer[self.KEY_hidden_return_seq]
            if return_sequences.lower() == "none":
                return_sequences = None
            elif return_sequences.lower() == "true":
                return_sequences = True
            else: return_sequences = False

            if "random" in layer[self.KEY_hidden_initializer].lower():
                kernel_start = initializers.random_normal
            elif "identity" in layer[self.KEY_hidden_initializer].lower():
                kernel_start = initializers.identity
            elif "none" in layer[self.KEY_hidden_initializer].lower():
                kernel_start = None
            elif "ones" in layer[self.KEY_hidden_initializer].lower():
                kernel_start = initializers.ones
            elif "uniform" in layer[self.KEY_hidden_initializer].lower():
                kernel_start = initializers.RandomUniform
            else: raise ValueError("This initializer is not recognized for this program.")

            if nn_type.upper() == "DENSE":
                hidden_lays = layers.Dense(unit, activation=activation_func,
                                                 kernel_initializer=kernel_start)
            elif nn_type.upper() == "LSTM":
                hidden_lays = layers.LSTM(unit, activation=activation_func, return_sequences=return_sequences, kernel_initializer=kernel_start)
            else:
                raise ValueError("This layer type is not available choose from: [Dense, LSTM]")
            yield hidden_lays



if __name__ == "__main__":

    src = LoadConfig()
    network_structure = src.config_file
    GenLayers = Layers(network_structure)


    model = Sequential()
    model.add(GenLayers.create_input_layer())
    for layer in GenLayers.generate_hidden_layer():
        model.add(layer)
    print(model.summary())

    # This applies to only OwnPred usage.
    # shuffle order is right for the showTrainTest method, but using pandas.concat,
    # the indexes are applied, so x_test,y_test must be resetted.
    # output = pandas.concat([NN.x_test.reset_index(inplace=False, drop=False),
    #                         NN.y_test.reset_index(inplace=False, drop=False), NN.preds], axis= 1)
    #
    # output.to_csv("result.csv")
