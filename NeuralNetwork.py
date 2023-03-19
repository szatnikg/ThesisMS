from tensorflow import keras, metrics
import numpy as np

import DataProcessing
import GenerateData
import pandas as pd
from matplotlib import pyplot as plt
import time
from pathlib import Path
import os
import json
from loader import Layers

class InputProcessing():

    def __init__(self, x=[], y=[],
                 OwnPred_x=[], OwnPred_y=[],
                 x_columns=["x"], y_columns=["y"]):
        # creating class variables
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Provide x and y value!")
        self.OwnPred_x = OwnPred_x.copy()
        self.OwnPred_y = OwnPred_y.copy()
        self.x = x
        self.y = y

        self.x_columns = x_columns
        self.y_columns = y_columns

        self.preprocessed = False

    def normalize_data(self, features= [], is_normalize=True, scale_type="normal", label_feature_name="y"):

        if not is_normalize:
            return
        # normalizing data by choosen features respectively
        self.scale_type = scale_type
        self.scale_x = DataProcessing.Scaler(features=features)
        self.x = self.scale_x.normalize(self.x,
                                     scale_type = self.scale_type,
                                     label_feature_name=label_feature_name,
                                     prediction_feature_name="preds")

        self.scale_y = DataProcessing.Scaler(features=features)
        self.y = self.scale_y.normalize(self.y,
                                     scale_type=self.scale_type,
                                     label_feature_name=label_feature_name,
                                     prediction_feature_name="preds")

        # Is it right normalizing Ownpred with x_train?
        self.OwnPred_x = self.scale_x.normalize(self.OwnPred_x,
                                        scale_type=self.scale_type,
                                        label_feature_name=label_feature_name,
                                        prediction_feature_name="preds")
        self.preprocessed = True
        return self.x, self.y, self.OwnPred_x, self.OwnPred_y

    def denormalize_data(self, x=None, y=None, OwnPred_x=None, OwnPred_y=None , preds=None, is_preds_normalized=True):
        # denormalizing features by the chosen features in method "normalize_data"

        if not self.preprocessed:
            return
            # raise ReferenceError("First call the preprocessing method with normalize=True")
        if x is not None:
            self.x = x
            self.y = y
            self.OwnPred_y = OwnPred_y
            self.OwnPred_x = OwnPred_x
        if preds is not None:
            self.preds = preds

        self.x = self.scale_x.denormalize(self.x, is_preds_normalized=False)
        self.y = self.scale_y.denormalize(self.y, is_preds_normalized=False)

        self.OwnPred_x = self.scale_x.denormalize(self.OwnPred_x, is_preds_normalized=False)

        if self.shuffle and len(self.OwnPred_x) > 0:
            self.x = self.x.iloc[self.shuffler]
            self.y = self.y.iloc[self.shuffler]

        # separating preds-data due to unequal record number with original dataset
        if is_preds_normalized:
            self.preds = self.scale_y.denormalize(self.preds, is_preds_normalized=is_preds_normalized)
        self.splitting()

        return self.x, self.y, self.OwnPred_x, self.OwnPred_y, self.preds

    def split_train_test(self, train_split=0.8, shuffle=False):
        self.train_split = train_split
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffler = np.random.permutation(len(self.x))
            self.x = self.x.iloc[self.shuffler]
            self.y = self.y.iloc[self.shuffler]

        self.splitting()
        self.n_features = len(self.x_train.columns)
        # print(len(self.x_train), len(self.y_train),":train dataset\n",
        #       len(self.x_test), len(self.y_test),":test dataset")
        return self.x_train, self.y_train, self.x_test, self.y_test

    def splitting(self):
        slicer = int((len(self.x) * self.train_split))
        self.x_train = self.x[:slicer]
        self.y_train = self.y[:slicer]

        if len(self.OwnPred_x) == 0:
            self.x_test = self.x[slicer:]
            self.y_test = self.y[slicer:]
        else:
            self.x_test = self.OwnPred_x
            self.y_test = self.OwnPred_y

    def convert_to_array(self):

        data_list = [self.x_train, self.x_test, self.y_train, self.y_test]

        converted_list = []
        counter = 0
        for elem in data_list:
            elem = elem.to_numpy()
            if counter < 2: elem = elem.reshape(len(elem), self.n_features)
            else: elem = elem.reshape(len(elem), 1)
            converted_list.append(elem)
            counter += 1

        self.x_train, self.x_test,\
        self.y_train, self.y_test = converted_list[0], converted_list[1], \
                                    converted_list[2], converted_list[3]

        return self.x_train, self.x_test, self.y_train, self.y_test

    def convert_to_df(self):
        self.x_train, self.y_train, self.x_test, self.y_test = pd.DataFrame(self.x_train, columns=self.x_columns), \
                                                               pd.DataFrame(self.y_train, columns=self.y_columns), \
                                                               pd.DataFrame(self.x_test, columns=self.x_columns), \
                                                               pd.DataFrame(self.y_test, columns=self.y_columns)

    def call_convert_to_timeseries(self, batch_size, sequence_length=None):
        if sequence_length:
            self.sequence_length = sequence_length
        if len(self.x_train) > self.sequence_length:
            self.fit_data = self.convert_to_timeseries(self.x_train, target=self.y_train, sequence_length=self.sequence_length, batch_size=batch_size)
        else: self.fit_data = None
        if len(self.x_test) > self.sequence_length:
            self.pred_data = self.convert_to_timeseries(self.x_test, target=None, sequence_length=self.sequence_length, batch_size=batch_size)
        else: self.pred_data = None
        return self.fit_data, self.pred_data

    @staticmethod
    def convert_to_timeseries(data, target=None, sequence_length=1, batch_size=1):

        output = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=target,
            sequence_length=sequence_length,
            sequence_stride=1,

            shuffle=False,
            batch_size=batch_size)
        return output

    def create_timeseries_deviation(self, sequence_length=1):

        # data_list = [self.y_train, self.y_test]
        # converted_list = []
        self.sequence_length = sequence_length

        for sequence_number in range(0, self.sequence_length-1):
            self.y_train = self.handle_timeseries_deviation(self.y_train)
        # converted_list.append(self.handle_timeseries_deviation(elem))
        # self.y_train, self.y_test = converted_list[0], converted_list[1]
        return self.y_train

    @staticmethod
    def handle_timeseries_deviation(np_range):

        original_shape = np_range.shape
        inserted_range = np.insert(np_range, len(np_range), 0)
        new_range = np.delete(inserted_range, 0) \
            .reshape(original_shape)
        return new_range


class NeuralNetwork(InputProcessing, Layers):

    def __init__(self, model_name,
                 x=[], y=[],
                 OwnPred_x=[], OwnPred_y=[],
                 network_structure= {},
                 x_columns=["x"], y_columns=["y"]):
        proj_folder = Path().absolute()  # alternative: Path(__file__).parent.resolve()
        self.data_folder = os.path.join(proj_folder, "..", "project_data")
        self.network_structure = network_structure
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        super().__init__(x=x, y=y, OwnPred_x=OwnPred_x, OwnPred_y=OwnPred_y, x_columns=x_columns, y_columns=y_columns)
        self.layer_generator = Layers.__init__(self, layer_obj=self.network_structure)
        self.model_name = model_name
        self.model_path = os.path.join(self.data_folder, self.model_name, self.model_name + ".h5")

    def showValLoss(self):
        hist = pd.DataFrame(self.history_model.history)
        #print(self.history_model.history["mse"][0])
        hist['epoch'] = self.history_model.epoch
        if self.classification:
            train_loss = 'acc'
        else:
            train_loss = 'loss'
        label_train = 'training_' + train_loss
        label_val = 'validation_' + train_loss
        plt.plot(self.history_model.history['loss'], label=label_train)
        plt.plot(self.history_model.history['val_loss'], label=label_val)
        plt.xlabel('Epoch')
        plt.ylabel('Error [unit]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def showTrainTest(self, with_pred=None, column_name="x"):

        plt.figure("Neural Network Performance",figsize=(10, 5))
        plt.style.use('bmh')
        if with_pred:
            plt.scatter(self.x_test[:len(self.preds)][column_name], self.preds["preds"], c='r', label='Predicted data', s=6)
        plt.scatter(self.x_train[column_name], self.y_train, c='b', label='Training data', s=6)
        # if len(self.OwnPred_x) == 0:
        plt.scatter(self.x_test[column_name], self.y_test, c='g', label='Testing data', s=5)
        plt.title(f"{self.model_name} "+"planed epoch = "+str(self.epoch)+f" Stopped at: {self.es}" )
        plt.xlabel(column_name)
        plt.ylabel("Evaluation feature")
        plt.legend()
        plt.show()
        # print("preds:", self.preds.shape, "\n", "test: ", self.y_test.shape)

    def build_model(self, nn_type="ann", loaded_model=False, classification=False):
        # building an RNN from scratch or loading an already existing model
        self.nn_type = nn_type
        self.classification = classification
        self.loaded_model = loaded_model
        self.model = keras.Sequential()
        if self.loaded_model:
            self.model = keras.models.load_model(self.model_path)
            print(self.model.summary())
            return

        # if network_structure (self.layer_obj in Layer class) provided
        # use Layers Child-Class to create model for config,
        # otherwise use built in structure.
        if self.layer_obj:
            inp = self.create_input_layer(self.n_features)
            self.model.add(inp)
            for hidden_layer in self.generate_hidden_layer():
                self.model.add(hidden_layer)
            return

        if self.nn_type == "ann":
            input_lay = keras.Input(shape=(self.x_train.shape[1],))
            # constructing NN architecture

            first_lay = keras.layers.Dense(64, activation='relu',
                                           kernel_initializer=keras.initializers.random_normal)
            hidden_lays = keras.layers.Dense(32, activation='relu',
                                             kernel_initializer=keras.initializers.random_normal)

            if self.classification:
                output_lay = keras.layers.Dense(1, activation="sigmoid")
            else:
                output_lay = keras.layers.Dense(1, input_shape=(1,), activation="linear")
            self.model.add(input_lay)
            self.model.add(first_lay)
            self.model.add(hidden_lays)
            self.model.add(output_lay)

        # nn_type == "rnn"
        else:

            self.model.add(keras.layers.LSTM(30, activation='relu', return_sequences=True, input_shape=(None, self.n_features)))
            self.model.add(keras.layers.LSTM(30, activation='relu', return_sequences=True))
            self.model.add(keras.layers.LSTM(30, activation='relu', return_sequences=False))
            self.model.add(keras.layers.Dense(1))


        # properties: print("weights: \n",self.model.weights)
        print(self.model.summary())

    def train_network(self,  epoch=90, batch_size=1, loss="mse", learning_rate=0.001,
                      earlystop=0, metrics=["mse"], further_training=True):
        self.epoch = epoch
        self.es = self.epoch

        if not self.loaded_model:
            # learning schedule for decaying learning_rate
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=20000,
                decay_rate=0.9)
            self.model.compile(loss=loss,  # mae stands for mean absolute error
                               optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # stochastic GD
                               metrics=metrics) # metrics=['accuracy']
        if further_training:
            # training the model
            if earlystop:
                EarlyStop = keras.callbacks.EarlyStopping(patience=earlystop)
            else: EarlyStop = []

            # NN training start time
            start = time.time()
            if self.nn_type == 'ann':
                self.history_model = self.model.fit(self.x_train, self.y_train, shuffle=True,
                                                    epochs=self.epoch,
                                                    batch_size=batch_size, verbose=0,
                                                    validation_split=0.2,
                                                    callbacks=[EarlyStop])
            else:
                self.history_model = self.model.fit(self.fit_data, shuffle=True,
                                                    epochs=self.epoch,
                                                    batch_size=batch_size, verbose=0,
                                                    callbacks=[EarlyStop])
            ## NN training time stop
            end = time.time()
            self.runtime = end - start

            if earlystop:
                self.es = EarlyStop.stopped_epoch
                if self.es != 0:
                    print("Stopped at epoch number:" ,str(self.es))
            else:
                self.es = self.epoch
            return

    def predictNN(self, pred_columns=["preds"]):
        if self.nn_type == 'ann': self.preds = self.model.predict(self.x_test, verbose=0)
        else: self.preds = self.model.predict(self.pred_data, verbose=0)
        try:
            self.preds = pd.DataFrame(self.preds, columns=pred_columns)
        except:
            np.squeeze(self.preds, axis=2)
            self.preds = pd.DataFrame(self.preds, columns=pred_columns)

    def evaluate(self):
        # This has to be called, after postprocessing (denormalization),
        # to match the y_test data with the (previously normalized, now denormalized) preds-data
        # ToDo get R^2 func or similar for evaluation...
        # print("x_test size:", len(self.x_test), "\n",
        #       "preds size ", len(self.preds), "\n",
        #       "y_test_size: ", len(self.y_test))

        if len(self.y_test) > 0:
            # get data about predictions
            # self.y_test.reset_index(inplace=True, drop=True)
            self.mae = metrics.mean_absolute_error(y_true=self.y_test[:len(self.preds)],
                                                 y_pred=self.preds).numpy()
            # metrics return a tf.Tensor object, so I convert to np-ndarray.
            mse = metrics.mean_squared_error(y_true=self.y_test[:len(self.preds)],
                                                y_pred=self.preds).numpy()
            if self.classification:
                acc = metrics.Accuracy()
                self.preds.loc[round(self.preds["preds"]) == 1 , 'correct_output'] = 1
                self.preds.loc[round(self.preds["preds"]) == 0, 'correct_output'] = 0
                acc_input = pd.DataFrame({"preds": self.preds["correct_output"]})
                acc.update_state(self.y_test, acc_input)
                print("accuracy: ",acc.result().np())
            # acc = metrics.RootMeanSquaredError()
            # acc.update_state(self.y_test, self.preds.squeeze())

            # print("mae:", self.mae, "\n",
            #       "mse:", mse,"\n")

    def save_model(self, model_lib=0):
        if not model_lib:
            model_lib = os.path.join(self.data_folder, self.model_name)
        if not os.path.exists(model_lib):
            os.mkdir(model_lib)
        self.model.save(os.path.join(model_lib, f"{self.model_name}.h5"))

        with open(os.path.join(model_lib, self.model_name + "_config.json"), "w") as json_output:
            json.dump(self.network_structure, json_output, indent=4)
        return model_lib

if __name__ == "__main__":

    pred_x = [] # own_pred_data.iloc[::, :-2]
    pred_y = [] #own_pred_data.iloc[::,-2]  #pd.DataFrame({"y": [j**2 for j in range(460, 560)]})
