from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
import leargist

plt.ion()
gist_size = 960


def DisplayPlot(train, valid, ylabel, number=0):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()
    plt.pause(0.0001)


def Save(fname, data):
    """Saves data to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def Load(fname):
    """Loads data from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))


def load_train():
    X_dirname = '../411a3/train'
    Y_filename = '../411a3/train.csv'
    X_filelist = image.list_pictures(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_train = np.zeros((7000, gist_size))
    y_train = Y_list[:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(7000):
        im = Image.open(X_filelist[i])
        descriptors = leargist.color_gist(im)
        X_train[i, :] = descriptors
        print('Load image: ' + X_filelist[i])

    return X_train, y_train


def load_val():
    X_dirname = '../411a3/val'
    X_filelist = image.list_pictures(X_dirname)

    X_val = np.zeros((970, gist_size))

    for i in range(970):
        im = Image.open(X_filelist[i])
        descriptors = leargist.color_gist(im)
        X_val[i, :] = descriptors
        print('Load image: ' + X_filelist[i])

    return X_val


def save_train():
    train_file = '../411a3/train.npz'
    X_train, y_train = load_train()
    train = {
        'X_train': X_train,
        'y_train': y_train
    }
    Save(train_file, train)
    print('Train data saved to: ' + train_file)


def save_val():
    val_file = '../411a3/val.npz'
    X_val = load_val()
    val = {
        'X_val': X_val
    }
    Save(val_file, val)
    print('Val data saved to: ' + val_file)


if __name__ == '__main__':
    save_train()
    save_val()
