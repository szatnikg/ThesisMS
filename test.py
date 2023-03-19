import os
from pathlib import Path
from main import NN_interface
from GenerateData import genUnNormalizedData, genSinwawe
import json
import shutil

class ConfigContainer:
    def __init__(self):
        pass

    def container_I(self):
        self.o = {
            "model_name": "qubic_prediction",
            "high_level_nn_type": "ann",
            "show_plot": 0,
            "epoch": 200,
            "batch_size": 10,
            "loaded_model": 0,
            "sequence_length": 1,
            "train_split": 0.76,
            "further_training": 1,
            "scale_type": "normal",
            "label_feature_name_for_normalization": "y",
            "show_column_name_in_plot": "x",
            "shuffle": 1,
            "want_to_normalize": 1,
            "model_lib": 0,

            "input_layer": {"type": "Dense",
                            "shape_1": "n_features",
                            "shape_2": "None"
                            },

            "hidden_layers": [{"type": "Dense",
                               "unit": 64,
                               "initializer": "random_normal",
                               "activation": "relu",
                               "return_sequences": "true"
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
        return self.o

    def container_II(self):
        self.o = {
            "model_name": "basic_prediction",
            "high_level_nn_type": "ann",
            "show_plot": 0,
            "epoch": 100,
            "batch_size": 4,
            "loaded_model": 0,
            "sequence_length": 1,
            "train_split": 0.76,
            "further_training": 1,
            "scale_type": "normal",
            "label_feature_name_for_normalization": "y",
            "show_column_name_in_plot": "x",
            "shuffle": 1,
            "want_to_normalize": 0,
            "model_lib": 0,

            "input_layer": {"type": "Dense",
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
        return self.o

    def container_III(self):
        self.o = {
    "model_name": "timeseries_prediction",
    "high_level_nn_type": "rnn",
    "epoch": 200,
    "batch_size": 10,
    "loaded_model": 0,
    "sequence_length": 4,
    "train_split": 0.76,
    "further_training": 1,
    "scale_type": "normal",
    "label_feature_name_for_normalization": "y",
    "show_column_name_in_plot": "x",
    "shuffle": 1,
    "want_to_normalize": 1,
    "show_plot": 0,
    "model_lib": 0,

    "input_layer": {"type": "LSTM",
                    "shape_1": "None",
                    "shape_2": "n_features"
                    },

    "hidden_layers": [ {  "type": "LSTM",
                "unit": 30 ,
                "initializer": "random_normal",
                "activation": "relu",
                "return_sequences": "true"
                  },
                    {  "type": "LSTM",
                "unit": 30 ,
                "initializer": "random_normal",
                "activation": "relu",
                "return_sequences": "false"
                   },
                    {"type": "Dense",
                "unit": 1 ,
                "initializer": "None",
                "activation": "None",
                "return_sequences": "None"
                   }
                    ]
    }
        return self.o

class Tester(ConfigContainer):

    def __init__(self):
        super().__init__()
        self.proj_folder = Path().absolute()  # alternative: Path(__file__).parent.resolve()
        self.test_folder = os.path.join(self.proj_folder, "test")

        # remove "test" folder under project and recreate it
        shutil.rmtree(self.test_folder, ignore_errors=True)
        os.mkdir(self.test_folder)

    def unit_test(self):
        # a unit test is defined here as a successful run with the given data
        # and NN structure + hyperparameters the completeness are measured by
        # the relative or absolute error for the model in the resulting dataframes
        # y_test -- preds , # self.should_perform is the performance indicator

        # generate data
        from GenerateData import genUnNormalizedData
        data_I = genUnNormalizedData(0, 800, type='square', step=1)
        data_II = genUnNormalizedData(0, 800, type='root', step=1)
        data_III = genSinwawe(2,1140)


        should_perform_list = [5, 6, 9]
        model_name_list = ["square_genData_reg_normalized",
                           "root_genData_reg_no_normalization",
                           "sinWawe_timeseries_normalized"]
        # modify config.json to test-specifications
        config_file_list = [self.container_I(), self.container_II(), self.container_III()]
        data_list = [data_I, data_II, data_III]

        for curr_param in range(len(model_name_list)):

            self.model_name = model_name_list[curr_param]
            data = data_list[curr_param]
            config_file = config_file_list[curr_param]
            should_perform = should_perform_list[curr_param]

            try:
                self.run(self.model_name, data, config_file)
                if self.performed_value < should_perform:
                    message = f"unit_test {curr_param+1}: {self.model_name}  ----  PASSED" \
                              + "\n" +"                 error %: " + str(round(self.performed_value, 2)) + " required: "+ str(should_perform)
                else:
                    print(f"unit_test {curr_param+1}: {self.model_name} error % too high! : ", str(round(self.performed_value, 2)), " required: ", str(should_perform))
                    raise AssertionError
            except:
                    message = f"unit_test {curr_param+1}: {self.model_name}  !!!!  FAILED "

            yield message

    def run(self, model_name, data, config_file):
        self.unit_lib = os.path.join(self.test_folder, model_name)

        x_columns = data.columns[:-1]
        x_columns = [col for col in x_columns]
        y_columns = data.columns[-1]
        if not type(y_columns) == str:
            y_columns = [col for col in y_columns]

        # modify config.json to test-specifications
        config_file["model_lib"] = self.unit_lib
        config_file["model_name"] = self.model_name
        self.apply_config(config_file)

        # build, run Neural Network
        self.tester_IF = NN_interface(data[x_columns], data[y_columns])
        # print("y-test:", self.tester_IF.NN.y_test)
        # print("preds:", self.tester_IF.NN.preds["preds"])
        x_test = self.tester_IF.NN.x_test.reset_index(drop=True)
        y_test = self.tester_IF.NN.y_test.reset_index(drop=True)

        comparision_df = self.tester_IF.compare_performance(x_test, y_test[self.tester_IF.label_feature_name],
                                                            self.tester_IF.NN.preds["preds"],
                                                            self.unit_lib, self.model_name)

        self.performed_value = comparision_df.sort_values("rel_error_percent", ignore_index=True)[
                               : int(len(comparision_df) * 0.8)].mean()
        self.performed_value = self.performed_value["rel_error_percent"]

        return None

    def apply_config(self, config_file):
        with open("config.json", "w") as output:
            json.dump(config_file, output)

if __name__ == "__main__":

    testing = Tester()
    for result in testing.unit_test():
        print(result)
