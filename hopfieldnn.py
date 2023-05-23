import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product


def hopfield_train(x_train, y_train, class_n):
    j = -1
    classes = list()
    train_unique = list()
    while len(classes) < class_n:
        j = j + 1
        if classes.__contains__(y_train[j]):
            continue
        else:
            classes.append(y_train[j])
            train_unique.append(x_train[j])

    # use pandas here
    for i in range(len(classes)):
        train_unique[i] = convert_binary(train_unique[i])

    for i in range(len(x_train)):
        x_train[i] = convert_binary(x_train[i])

    train_unique = np.asarray(train_unique)

    train_unique = train_unique.reshape(train_unique.shape[0], -1)
    x_train = x_train.reshape(x_train.shape[0], -1)

    hopfield = HopfieldNetwork(train_unique, rule="pseudo-inverse")
    pred = hopfield.predict(x_train)

    predicted = 0
    for i in range(len(x_train)):
        class_index = classes.index(y_train[i])
        if np.array_equal(train_unique[class_index], pred[i]):
            predicted += 1
    accuracy = predicted / len(y_train)

    n_examples = 10
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 4))
    indices = np.random.choice(x_train.shape[0], n_examples)
    for i in range(n_examples):
        axs[0, i].imshow(x_train[indices[i]].reshape(28, 28), cmap='gray')
        axs[1, i].imshow(pred[indices[i]].reshape((28, 28)), cmap='gray')
    plt.show()

    return accuracy


def convert_binary(arr):
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            if arr[j, k] < 0.5:
                arr[j, k] = -1
            else:
                arr[j, k] = 1
    return arr


class HopfieldNetwork(object):
    def __init__(self, pattern, rule='pseudo-inverse'):
        self.n = pattern[0].size
        self.order = np.arange(self.n)

        if rule == 'hebbian':
            self.w = np.tensordot(pattern, pattern, axes=((0), (0))) / len(pattern)
            self.w[self.order, self.order] = 0.0
        elif rule == 'pseudo-inverse':
            c = np.tensordot(pattern, pattern, axes=((1), (1))) / len(pattern)
            cinv = np.linalg.inv(c)
            self.w = np.zeros((self.n, self.n))
            for k, l in product(range(len(pattern)), range(len(pattern))):
                self.w = self.w + cinv[k, l] * pattern[k] * pattern[l].reshape((-1, 1))
            self.w = self.w / len(pattern)

    def predict(self, x, iters=4):
        h = np.array(x, dtype=float)
        for _ in range(iters):
            np.random.shuffle(self.order)
            for i in self.order:
                h[:, i] = np.tensordot(self.w[i], h, axes=((-1), (-1)))
                h[:, i] = np.where(h[:, i] < 0, -1.0, 1.0)
        return h
