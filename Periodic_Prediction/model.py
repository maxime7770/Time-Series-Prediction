import numpy as np
import math
import random
from math import sin, cos, pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from generation import (
    sequence_len,
    features_len,
    train_generator,
    test_generator,
    epochs,
    fit_verbosity,
)

epochs = 5

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(sequence_len, features_len)))
# model.add( keras.layers.GRU(200, dropout=.1, recurrent_dropout=0.5, return_sequences=False, activation='relu') )
model.add(keras.layers.GRU(20,
                           dropout=0.1,
                           recurrent_dropout=0,
                           return_sequences=True,
                           recurrent_activation="sigmoid",
                           activation="tanh"))
model.add(keras.layers.GRU(20,
                           dropout=0.1,
                           recurrent_dropout=0,
                           return_sequences=False,
                           recurrent_activation="sigmoid",
                           activation="tanh"))
model.add(keras.layers.Dense(features_len))

model.summary()


model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])


history = model.fit(
    train_generator,
    epochs=epochs,
    verbose=fit_verbosity,
    validation_data=test_generator,
)

model.save("model.h5")

# plt.plot(history.history["acc"])
# plt.plot(history.history["val_acc"])
# plt.title("model accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(["train", "validation"], loc="upper left")
# plt.show()
