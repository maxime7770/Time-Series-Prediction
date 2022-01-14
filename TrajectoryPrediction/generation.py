import numpy as np
import math, random
from math import sin, cos, pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# parameters to be used


max_t = 10000
delta_t = 0.02
features_len = 2

sequence_len = 100
predict_len = 5

scale = 1
train_prop = 0.8
batch_size = 32
epochs = 5
fit_verbosity = 1


def traj_init(s=122):

    if s > 0:
        random.seed(s)
    traj_init.params_x = [random.gauss(0.0, 1.0) for u in range(8)]
    traj_init.params_y = [random.gauss(0.0, 1.0) for u in range(8)]


def move(t):
    k = 0.5
    [ax1, ax2, ax3, ax4, kx1, kx2, kx3, kx4] = traj_init.params_x
    [ay1, ay2, ay3, ay4, ky1, ky2, ky3, ky4] = traj_init.params_y

    x = (
        ax1 * sin(t * (kx1 + 20))
        + ax2 * cos(t * (kx2 + 10))
        + ax3 * sin(t * (kx3 + 5))
        + ax4 * cos(t * (kx4 + 5))
    )
    y = (
        ay1 * cos(t * (ky1 + 20))
        + ay2 * sin(t * (ky2 + 10))
        + ay3 * cos(t * (ky3 + 5))
        + ay4 * sin(t * (ky4 + 5))
    )

    return x, y


# get positions

traj_init(s=16)
x, y = 0, 0
positions = []
for t in np.arange(0.0, max_t, delta_t):
    positions.append([x, y])
    x, y = move(t)

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

if __name__ == "__main__":
    plt.plot([x[0] for x in x_train[:1000]], [x[1] for x in x_train[:1000]])
    plt.show()

    k1, k2 = sequence_len, predict_len
    i = random.randint(0, len(x_test) - k1 - k2)
    j = i + k1

    plt.plot([x[0] for x in x_test[i : j + k2]], [x[1] for x in x_test[i : j + k2]])
    plt.plot([x[0] for x in x_test[j : j + k2]], [x[1] for x in x_test[j : j + k2]])
    plt.show()

    x, y = train_generator[0]
    print(f"Number of batch trains available : ", len(train_generator))
    print("batch x shape : ", x.shape)
    print("batch y shape : ", y.shape)

