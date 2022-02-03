import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
import math
import random
import matplotlib.pyplot as plt

import pandas as pd

from preprocessing import close_position


sequence_len = 50
batch_size = 1
epochs = 5
fit_verbosity = 1

data = close_position
p = int(0.8 * data.shape[0])
train = data[:p]
test = data[p:]
print(train)
mean = train.mean()
std = train.std()
min = train.min()
max = train.max()

# train = (train - mean) / std
# test = (test - mean) / std

train = (train - min) / (max - min)
test = (test - min) / (max - min)
# print(train)

train = train.to_numpy()
test = test.to_numpy()


train_generator = TimeseriesGenerator(
    train, train, length=sequence_len, batch_size=batch_size
)
test_generator = TimeseriesGenerator(
    test, test, length=sequence_len, batch_size=batch_size
)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(sequence_len, 1)))

model.add(
    keras.layers.GRU(
        70,
        dropout=0.1,
        recurrent_dropout=0,
        return_sequences=False,
        recurrent_activation="sigmoid",
        activation="tanh",
    )
),
model.add(keras.layers.Dense(1))
model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])


history = model.fit(
    train_generator,
    epochs=epochs,
    verbose=fit_verbosity,
    validation_data=test_generator,
)

model.save("model.h5")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()
