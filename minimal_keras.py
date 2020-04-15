# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback

import tensorflow as tf

from utils import load_data, one_hot_encode

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default="./", help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

data_folder = args.data_folder

import gzip
import numpy as np
import struct


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]

print('training dataset is stored here:', data_folder)

X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0

y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
n_epochs = 20
batch_size = args.batch_size
learning_rate = args.learning_rate

y_train = one_hot_encode(y_train, n_outputs)
y_test = one_hot_encode(y_test, n_outputs)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
# first hidden layer
model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
# second hidden layer
model.add(Dense(n_h2, activation='relu'))
# output layer
model.add(Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])



history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
print('Test loss:', score[0])

print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)


# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
