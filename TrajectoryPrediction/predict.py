import numpy as np
import math, random
from math import sin, cos, pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from generation import x_test, sequence_len


model = keras.models.load_model("model2.h5")


s = random.randint(0, len(x_test) - sequence_len)

sequence = x_test[s : s + sequence_len]
sequence_true = x_test[s : s + sequence_len + 1]
sequence_pred = np.append(sequence, model.predict(np.array([sequence])), axis=0)
plt.plot(
    [x[0] for x in sequence_true],
    [x[1] for x in sequence_true],
    color="blue",
    label="true",
)
plt.plot(
    [x[0] for x in sequence_pred],
    [x[1] for x in sequence_pred],
    color="red",
    label="predicted",
)
plt.legend()
plt.show()


def get_prediction(dataset, model, iterations=4):

    # ---- Initial sequence
    #
    s = random.randint(0, len(dataset) - sequence_len - iterations)
    sequence = dataset[s : s + sequence_len]
    sequence_pred = sequence.copy()
    sequence_true = dataset[s : s + sequence_len + iterations].copy()

    # ---- Iterate
    #
    sequence_pred = list(sequence_pred)

    for i in range(iterations):
        sequence = sequence_pred[-sequence_len:]
        prediction = model.predict(np.array([sequence]))
        sequence_pred.append(prediction[0])

    # ---- Extract the predictions
    #
    # prediction = np.array(sequence_pred[-iterations:])

    return sequence_true, sequence_pred


sequence_true, sequence_pred = get_prediction(x_test, model, iterations=30)

plt.plot(
    [x[0] for x in sequence_true],
    [x[1] for x in sequence_true],
    color="blue",
    label="true",
)
plt.plot(
    [x[0] for x in sequence_pred],
    [x[1] for x in sequence_pred],
    color="red",
    label="predicted",
)
plt.legend()
plt.show()

