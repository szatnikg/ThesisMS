import array

import pandas
from tensorflow import keras, metrics
import numpy

import DataProcessing
import GenerateData
import pandas as pd
from matplotlib import pyplot as plt
from math import ceil
import time



class sample_NN():
    def __init__(self, model_name):
        self.model_name = model_name
        self.preprocessed = False

    def preprocessing(self, x, y, normalize=True,OwnPred_x=[],OwnPred_y=[] ,features=[],label_feature_name="y"):
        # Todo implement features_x, features_y
        #  .reset_index(inplace=True, drop=True) somewhere needed for handling output dataframe record inorder.
        self.OwnPred_x = OwnPred_x.copy()
        self.OwnPred_y = OwnPred_y.copy()
        if normalize:
            self.scale_x = DataProcessing.Scaler(features=features)
            self.x = self.scale_x.normalize(x,
                                         label_feature_name=label_feature_name,
                                         prediction_feature_name="preds")

            self.scale_y = DataProcessing.Scaler(features=features)
            self.y = self.scale_y.normalize(y,
                                         label_feature_name=label_feature_name,
                                         prediction_feature_name="preds")

            # Is it right normalizing Ownpred with x_train?
            self.OwnPred_x = self.scale_x.normalize(self.OwnPred_x,
                                            label_feature_name=label_feature_name,
                                            prediction_feature_name="preds")
            self.preprocessed = True
        else:
            self.x = x
            self.y = y

    def postprocessing(self, is_preds_normalized=True):
        if not self.preprocessed:
            return
            # raise ReferenceError("First call the preprocessing method with normalize=True")

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

    def splitting(self):
        slicer = ceil((len(self.x) * self.train_split))
        self.x_train = self.x[:slicer]
        self.y_train = self.y[:slicer]

        if len(self.OwnPred_x) == 0:
            self.x_test = self.x[slicer-1:]
            self.y_test = self.y[slicer-1:]
        else:
            self.x_test = self.OwnPred_x
            self.y_test = self.OwnPred_y

    def split_train_test(self, shuffle=True, train_split=0.8):
        self.shuffle = shuffle
        self.train_split = train_split

        if shuffle:
            self.shuffler = numpy.random.permutation(len(self.x))
            self.x = self.x.iloc[self.shuffler]
            self.y = self.y.iloc[self.shuffler]
            # resetting and dropping old indexes can be done by:
            # my_df.reset_index(inplace=True, drop=True)

        self.splitting()
        print(len(self.x_train), len(self.y_train),":train dataset\n",
              len(self.x_test), len(self.y_test),":test dataset")

        return

    def showValLoss(self):
        hist = pd.DataFrame(self.history_model.history)
        #print(self.history_model.history["mse"][0])
        hist['epoch'] = self.history_model.epoch
        if self.classification:
            train_loss = 'acc'
            val_loss = 'val_acc'
        else:
            train_loss = 'loss'
            val_loss = 'val_loss'
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
        plt.figure("Neural Network Performance",figsize=(12, 6))
        if with_pred:
            plt.scatter(self.x_test[column_name], self.preds["correct_output"], c='r', label='Predicted data')
        plt.scatter(self.x_train[column_name], self.y_train, c='b', label='Training data')
        if len(self.OwnPred_x) == 0:
            plt.scatter(self.x_test[column_name], self.y_test, c='g', label='Testing data')
        plt.title(f"{self.model_name} "+"planed epoch = "+str(self.epoch)+f" Stopped at: {self.es}" )
        plt.xlabel(column_name)
        plt.ylabel("Evaluation feature")
        plt.legend()
        plt.show()
        print("preds:",self.preds,"\n","test: ",self.y_test)

    def build_model(self, nn_type="simple", loaded_model=False, classification=False):
        # building a NN from scratch or loading an already existing model
        self.classification = classification
        if loaded_model:
            # It can be used to reconstruct the model identically.
            self.model = keras.models.load_model(f"{self.model_name}.h5")

            print([layer2.name for layer2 in self.model.layers])
            print("loaded model summary: ",self.model.summary())
            return

        if nn_type=="simple":

            # constructing NN architecture
            input_lay = keras.Input(shape=(self.x_train.shape[1],))
            first_lay = keras.layers.Dense(16, activation=keras.activations.relu,
                kernel_initializer=keras.initializers.random_normal)
            hidden_lays = keras.layers.Dense(32, activation=keras.activations.relu,
                kernel_initializer=keras.initializers.random_normal)

            if self.classification:
                output_lay = keras.layers.Dense(1, activation="sigmoid")
            else:
                output_lay = keras.layers.Dense(1, input_shape=(1,), activation="linear")

            # building the NN network
            self.model = keras.Sequential()
            #self.model.add(keras.layers.Dropout(0.2, input_shape=(1,)))
            #self.model.add(keras.layers.BatchNormalization())
            self.model.add(input_lay)
            self.model.add(first_lay)
            self.model.add(hidden_lays)
            self.model.add(output_lay)

            # print(self.model.summary())
            # print("weights: \n",self.model.weights)

        elif nn_type=="SimpleRNN":
            self.model = keras.Sequential()
            self.model.add(keras.layers.SimpleRNN(20, return_sequences=True, input_shape=(1,)))
            self.model.add.keras.layers.Dense(1)

    def network_specs(self,  epoch=90, batch_size=1,loss="mse", learning_rate=0.2, earlystop=0, metrics=["mse"]):
        self.epoch = epoch

        # single_feature_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
        # single_feature_normalizer.adapt(self.x_test)

        # learning schedule for decaying learning_rate
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9)
        self.model.compile(loss=loss,  # mae stands for mean absolute error
                      optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)  # stochastic GD
                      ,metrics=metrics) # metrics=['accuracy']

        # training the model
        if earlystop:
            EarlyStop = keras.callbacks.EarlyStopping(patience=earlystop)
        else: EarlyStop = []

        # NN training start time
        start = time.time()
        self.history_model = self.model.fit(self.x_train, self.y_train, shuffle=True,
                       epochs=self.epoch,
                       batch_size=batch_size,verbose=0,
                       validation_split=0.2,
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

    def predictNN(self, pred_columns=["preds"] ):
        self.preds = self.model.predict([self.x_test])
        self.preds = pandas.DataFrame(self.preds, columns=pred_columns)

    def evaluate(self):
        # This has to be called, after postprocessing (denormalization),
        # to match the y_test data with the (previously normalized, now denormalized) preds-data
        # ToDo get R^2 func or similar for evaluation...
        if len(self.y_test) > 0:
            # get data about predictions
            # self.y_test.reset_index(inplace=True, drop=True)
            self.mae = metrics.mean_absolute_error(y_true=self.y_test,
                                                 y_pred=self.preds).numpy()
            # metrics return a tf.Tensor object, so I convert to numpy-ndarray.
            mse = metrics.mean_squared_error(y_true=self.y_test,
                                                y_pred=self.preds).numpy()
            # ToDo implement Classification accuracy
            if self.classification:
                acc = metrics.Accuracy()
                self.preds.loc[round(self.preds["preds"]) == 1 , 'correct_output'] = 1
                self.preds.loc[round(self.preds["preds"]) == 0, 'correct_output'] = 0
                acc_input = pd.DataFrame({"preds": self.preds["correct_output"]})
                acc.update_state(self.y_test, acc_input)
                print("accuracy: ",acc.result().numpy())
            # acc = metrics.RootMeanSquaredError()
            # acc.update_state(self.y_test, self.preds.squeeze())

            # print("mae:", self.mae, "\n",
            #       "mse:", mse,"\n")

    def save_model(self):
        self.model.save(f"{self.model_name}.h5")

if __name__ == "__main__":
    # type(data) = pd.Dataframe
    data = pandas.read_excel(
        "C:\Egyetem\Diplomamunka\data\TanulokAdatSajat.xlsx")
    # GenerateData.genUnNormalizedData(500, type='square')
    # own_pred_data = pandas.read_excel("C:\Egyetem\Diplomamunka\data\TanulokAdatSajat_ownpred.xlsx")

    pred_x = [] # own_pred_data.iloc[::, :-2]
    pred_y = [] #own_pred_data.iloc[::,-2]  #pandas.DataFrame({"y": [j**2 for j in range(460, 560)]})



    NN = sample_NN("TryoutBasic")
    # Preprocess
    x = data[data.columns[:-2]]
    y = data[data.columns[-1]]
    NN.preprocessing(x,
                     y, normalize=True, features=[],
                     OwnPred_x=pred_x, OwnPred_y=pred_y,
                     label_feature_name="Teljesitmeny_jegy")
    NN.split_train_test(train_split=0.7, shuffle=True)

    NN.build_model(classification=True)
    NN.network_specs(epoch=100,
                     batch_size=50,
                     learning_rate=0.0035,
                     earlystop=40,
                     metrics=["accuracy"],
                     loss="binary_crossentropy")
    print("Training Time:",
          str(round(NN.runtime, 4)), "seconds."
          )
    NN.predictNN()
    NN.showValLoss()
    NN.postprocessing(is_preds_normalized=True)
    NN.evaluate()
    NN.showTrainTest(with_pred=True, column_name=data.columns[0])


    # shuffle order is right for the showTrainTest method, but using pandas.concat,
    # the indexes are applied, so x_test,y_test must be resetted.
    output = pandas.concat([NN.x_test.reset_index(inplace=False, drop=False),
                            NN.y_test.reset_index(inplace=False, drop=False), NN.preds], axis= 1)

    output.to_csv("result.csv")
    # NN.save_model(be_mae= 99999)