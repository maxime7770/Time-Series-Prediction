<<<<<<< HEAD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
import math, random
import matplotlib.pyplot as plt

import pandas as pd
from preprocessing import close_position

sequence_len = 50

data = close_position
p = int(0.8 * data.shape[0])
train = data[:p]
test = data[p:]
mean = train.mean()
std = train.std()
min = train.min()
max = train.max()

train = (train - min) / (max - min)
test = (test - min) / (max - min)
# print(train)

# train = (train - mean) / std
# test = (test - mean) / std

train = train.to_numpy()
test = test.to_numpy()


model = keras.models.load_model("model.h5")


def denormalize(mean, std, seq):
    nseq = std * seq + mean
    return nseq


def denormalize2(min, max, seq):
    nseq = (max - min) * seq + min
    return nseq


i = random.randint(0, len(test) - sequence_len)
sequence = test[i : i + sequence_len]
sequence_true = test[i : i + sequence_len + 1]


pred = model.predict(np.array([sequence]))
print(pred)
sequence = denormalize2(min, max, sequence)
sequence_true = denormalize2(min, max, sequence_true)
pred = denormalize2(min, max, pred)
print(sequence)
print(sequence_true)

sequence_pred = np.append(sequence, pred[0], axis=0)
print(sequence_pred)

plt.plot(
    sequence_true, color="blue", label="true",
)
plt.plot(
    sequence_pred, color="red", label="predicted",
)
plt.legend()
plt.show()

=======
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np
import math, random
import matplotlib.pyplot as plt

import pandas as pd
from preprocessing import close_position

sequence_len = 50

data = close_position
p = int(0.8 * data.shape[0])
train = data[:p]
test = data[p:]
mean = train.mean()
std = train.std()
min = train.min()
max = train.max()

train = (train - min) / (max - min)
test = (test - min) / (max - min)
# print(train)

# train = (train - mean) / std
# test = (test - mean) / std

train = train.to_numpy()
test = test.to_numpy()


model = keras.models.load_model("model.h5")


def denormalize(mean, std, seq):
    nseq = std * seq + mean
    return nseq


def denormalize2(min, max, seq):
    nseq = (max - min) * seq + min
    return nseq


i = random.randint(0, len(test) - sequence_len)
sequence = test[i : i + sequence_len]
sequence_true = test[i : i + sequence_len + 1]


pred = model.predict(np.array([sequence]))
print(pred)
sequence = denormalize2(min, max, sequence)
sequence_true = denormalize2(min, max, sequence_true)
pred = denormalize2(min, max, pred)
print(sequence)
print(sequence_true)

sequence_pred = np.append(sequence, pred[0], axis=0)
print(sequence_pred)

plt.plot(
    sequence_true, color="blue", label="true",
)
plt.plot(
    sequence_pred, color="red", label="predicted",
)
plt.legend()
plt.show()

>>>>>>> 68a3b1e894bc20852fef2f9f78e8d7e5ae7e00a4
