import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def choose_representative_samples(x, y, class_n):
    class_n = class_n
    classes = list()
    train_unique = list()
    for i in range(class_n):
        one_class_imgs = x[y == i]
        classes.append(i)
        train_unique.append(one_class_imgs[int(np.floor(np.random.random() * len(one_class_imgs)))])

    # plotting chosen representatives
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, class_n, figsize=(10, 4))
    for i in range(class_n):
        axs[i].imshow(train_unique[i].reshape(28, 28), cmap='gray')
    plt.show()

    return train_unique, classes


def hopfield_train(classes, train, x_test, y_test):
    # use pandas here
    for i in range(len(classes)):
        train[i] = convert_binary(train[i])

    for i in range(len(x_test)):
        x_test[i] = convert_binary(x_test[i])

    train = np.asarray(train)

    train = train.reshape(train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    hopfield = HopfieldNetwork(train, rule="pseudo-inverse")
    pred = hopfield.predict(x_test)

    predicted = 0
    for i in range(len(x_test)):
        class_index = classes.index(y_test[i])
        if np.array_equal(train[class_index], pred[i]):
            predicted += 1
    accuracy = predicted / len(y_test)

    n_examples = 10
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 4))
    indices = np.random.choice(x_test.shape[0], n_examples)
    for i in range(n_examples):
        axs[0, i].imshow(x_test[indices[i]].reshape(28, 28), cmap='gray')
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
