import tensorflow as tf
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

    def split_train_test(self, my_data, shuffle=True, train_split= 0.8,OwnPrediction_x=[],OwnPrediction_y=[]):
        self.x = numpy.array(my_data["x"])
        self.y = numpy.array(my_data["y"])
        if shuffle:
            shuffler = numpy.random.permutation(len(self.x))
            self.x = self.x[shuffler]
            self.y = self.y[shuffler]

        slicer = ceil((len(self.x)*train_split))
        self.x_train = self.x[:slicer+1]
        self.y_train = self.y[:slicer+1]
        if len(OwnPrediction_x) == 0:
            self.x_test = self.x[slicer:]
            self.y_test = self.y[slicer:]
        else:
            self.x_test = numpy.array(OwnPrediction_x)
            self.y_test = numpy.array(OwnPrediction_y)
        print(len(self.x_train),len(self.y_train),":train dataset\n",
              len(self.x_test),len(self.y_test),":test dataset")
        return
    def showValLoss(self):
        hist = pd.DataFrame(self.history_model.history)
        #print(self.history_model.history["mse"][0])
        hist['epoch'] = self.history_model.epoch
        plt.plot(self.history_model.history['loss'], label='training_loss')
        plt.plot(self.history_model.history['val_loss'], label='validation_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [unit]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def soft_acc(self):
        return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(self.y_test), tf.keras.backend.round(self.preds.squeeze())))

    # Usage: model.compile(..., metrics=[soft_acc])

    def showTrainTest(self,with_pred=None):
        plt.figure(figsize=(12, 6))
        if with_pred:
            plt.scatter(self.x_test, self.preds, c='r', label='Predicted data')
        plt.scatter(self.x_train, self.y_train, c='b', label='Training data')
        plt.scatter(self.x_test, self.y_test, c='g', label='Testing data')
        plt.title(f"{self.model_name} "+"planed epoch = "+str(self.epoch)+f" Stopped at: {self.es}" )
        plt.legend()
        plt.show()

    def build_model(self, nn_type="SimpleNN"):
        if nn_type=="SimpleNN":
            one_lay = tf.keras.layers.Dense(40, kernel_initializer='normal',
                                                 activation=tf.keras.activations.relu, input_shape=(1,))

            self.model = tf.keras.Sequential()
            #self.model.add(tf.keras.layers.Dropout(0.2, input_shape=(1,)))
            self.model.add(one_lay)
            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Dense(40, kernel_initializer='normal',
                                                 activation=tf.keras.activations.relu))

            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Dense(1, input_shape=(1,),activation="linear"))

            print(self.model.summary())
        elif nn_type=="SimpleRNN":
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=(1,)))
            self.model.add.tf.keras.layers.Dense(1)

    def implNN(self, loaded_model=False, epoch=90, batch_size=1, learning_rate=0.2, nn_type="simpleNN", earlystop=0):
        self.epoch = epoch

        if loaded_model:
            # It can be used to reconstruct the model identically.
            self.model = tf.keras.models.load_model(f"{self.model_name}.h5")
            for layer in self.model.layers:
                print([layer2.name for layer2 in self.model.layers])
            print("loaded model summary: ",self.model.summary())
            return
        # single_feature_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
        # single_feature_normalizer.adapt(self.x_test)
        self.build_model(nn_type=nn_type)

        #loss = ["mae", "categorical_crossentropy"]
        # learning schedule for decaying learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9)

        self.model.compile(loss=tf.keras.losses.mse,  # mae stands for mean absolute error
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)  # stochastic GD
                      ,metrics=["mse"]) # metrics=['accuracy']

        #training the model
        if earlystop:
            EarlyStop = tf.keras.callbacks.EarlyStopping(patience=30)
        self.history_model = self.model.fit(self.x_train, self.y_train, shuffle=True,
                       epochs=self.epoch,
                       batch_size=batch_size,verbose=0,
                       validation_split=0.2,
                       callbacks=[EarlyStop])
        self.es = EarlyStop.stopped_epoch
        if self.es != 0:
            print("Stopped at epoch number:" ,str(self.es))
        else:
            self.es = self.epoch

    def predictNN(self):
        self.preds = self.model.predict([self.x_test])
        # ToDo get R^2 func or similar for evaluation...
        # get data about predictions
        self.mae = tf.metrics.mean_absolute_error(y_true=self.y_test,
                                             y_pred=self.preds.squeeze()).numpy()
        mse = tf.metrics.mean_squared_error(y_true=self.y_test,
                                            y_pred=self.preds.squeeze()).numpy()
        # ToDo implement Classification accuracy
        if "classificationProblem":
            acc = tf.metrics.Accuracy()
            acc.update_state([[3.1,45]], [[3.0,45 ]])
            print("accuracy: ",acc.result().numpy())
        # acc = tf.metrics.RootMeanSquaredError()
        # acc.update_state(self.y_test, self.preds.squeeze())
        print("mae:", round(self.mae,5), "\n",
              "mse:", mse  ,"\n")

    def save_model(self, be_mae= 1):
        if self.mae <= be_mae:
            self.model.save(f"{self.model_name}.h5")

if __name__ == "__main__":
    pred_x = []#[0.95, 1,1.1,1.2,1.3,1.4,1.5]
    pred_y = []#[0.95**2, 1.002,1.1**2, 1.2**2+0.02,1.3**2 ,1.4**2+0.02, 1.5**2]

    data = GenerateData.genUnNormalizedData(700 , type='square')
    # data = getComplexData(1100)

    #Preprocess
    data = DataProcessing.Scaler().normalize(data, features=[])
    start = time.time()
    NN = sample_NN("TryoutBasic")
    NN.split_train_test(data, train_split=0.7, shuffle=True, OwnPrediction_x=pred_x,OwnPrediction_y=pred_y)
    NN.implNN(loaded_model=False, epoch=90, batch_size=5, learning_rate=0.0035, nn_type= "SimpleNN",earlystop=1)
    NN.predictNN()
    end = time.time()
    runtime = end-start
    print("Training Time:",
          str(round(runtime, 4)), "seconds."
          )
    NN.showValLoss()
    NN.showTrainTest(with_pred=True)

    # NN.save_model(be_mae= 99999)
    # linearPlot(data)


# layer parameters:
# layer = layers.Dense(
#     units=64,
#     kernel_initializer=initializers.RandomNormal(stddev=0.01),
#     bias_initializer=initializers.Zeros(),
#     activation=tf.keras.activations.relu,
#     input_shape=(1,)
# )