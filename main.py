# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import datetime
from feedforwardnn import feedforward_train

# Load the MNIST dataset
data = mnist.load_data()
j = -1
classes = list()
train = list()
while len(classes) < 9 and j < 200:
    j = j + 1
    if classes.__contains__(data[0][1][j]):
        continue
    else:
        classes.append(data[0][1][j])
        train.append(data[0][0][j])

for i in range(len(classes)):
    for j in range(len(train[0])):
        for k in range(len(train[1])):
            if train[i][j, k] < 128:
                train[i][j, k] = 0
            else:
                train[i][j, k] = 255

train = np.asarray(train)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = (X_train / 255).astype(float)
X_test = (X_test / 255).astype(float)
train = (train / 255).astype(float)
# Flatten the input images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
train = train.reshape(train.shape[0], -1)



# Define the Hopfield neural network model
model_hopfield = keras.Sequential([
    keras.layers.Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='sigmoid', trainable=False)
])
# Train the Hopfield network
model_hopfield.compile(optimizer='adam', loss='mse')
model_hopfield.fit(X_train, y_train, epochs=100)

# Feedforward network
model_feedforward = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2, name='layers_dropout'),
    keras.layers.Dense(10, activation='softmax')
])
logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(histogram_freq=1, log_dir=logdir)
# Compile model
model_feedforward.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


y_pred = model_feedforward.predict(X_test)
pred_labels = np.argmax(y_pred, axis=1)
# Show some examples of input and output images
n_examples = 5
fig, axs = plt.subplots(2, n_examples, figsize=(10, 4))
indices = np.random.choice(X_test.shape[0] ,n_examples)
for i in range(n_examples):
    axs[0, i].imshow(X_test[indices[i]].reshape(28, 28), cmap='gray')
    axs[1, i].imshow(y_pred[indices[i]].reshape(28, 28), cmap='gray')
plt.show()

model_feedforward.evaluate(X_test, y_test)

# TBD: trzeba zrobic zgodnie z poniższym repo, dla sieci Hopfielda mamy zbiegać do jednego z treningowych 10 numerków, dla sieci feedforward klasyfikacja, bo nie widzę innej opcji
# https://github.com/ccd97/hello_nn/blob/master/Hopfield-Network/code/np_hnn_reconstruction.py

train_data = [np.array(d) for d in {"5": train[0], "0": train[1]}.values()][:2]

a = train(train.shape[1], train)

def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    return w
np.outer()

# Function to test the network
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data

# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, data, steps=10):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res