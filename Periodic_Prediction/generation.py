import numpy as np
import math
import random
from math import sin, cos, pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# parameters to be used

max_t = 10000
delta_t = 0.1
features_len = 1

sequence_len = 500
predict_len = 10

scale = 1
train_prop = 0.8
batch_size = 32
epochs = 10
fit_verbosity = 1


positions = [np.sin(x) for x in np.arange(0.0, max_t, delta_t)]
# rescaled dataset

n = int(len(positions) * scale)
dataset = np.array(positions[:n])

k = int(len(dataset) * train_prop)
x_train = dataset[:k]
x_test = dataset[k:]

# normalization

mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

print("Dataset generated.")
print("Train shape is : ", x_train.shape)
print("Test  shape is : ", x_test.shape)


train_generator = TimeseriesGenerator(
    x_train, x_train, length=sequence_len, batch_size=batch_size
)
test_generator = TimeseriesGenerator(
    x_test, x_test, length=sequence_len, batch_size=batch_size
)
