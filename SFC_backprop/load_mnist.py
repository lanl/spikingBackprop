#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import mnist
import numpy as np
import skimage.transform


def rescale(data, sidelen=28):
    scaled = []
    for i in np.arange(data.shape[0]):
        s = skimage.transform.resize(data[i], (sidelen, sidelen), anti_aliasing=False, mode='constant')
        scaled.append(s)
    return np.asarray(scaled).reshape(-1, sidelen * sidelen)


def train_images():
    return mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz', force=True)


def test_images():
    return mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', force=True)


def train_labels():
    return mnist.download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', force=True)


def test_labels():
    return mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', force=True)


def load_mnist(sidelen=20, crop=4, shuffle=True,
               url='https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/', dataset_name='mnist'):
    """
    First cropping on all sides, then rescaling to sidelen x sidelen.
    Args:
        sidelen:
        crop:
    """
    try:
        with np.load('./' + dataset_name + str(sidelen) + '_' + str(crop) + '.npz') as mnist_data:
            mnist_train_data = mnist_data['train_data']
            mnist_test_data = mnist_data['test_data']
            mnist_train_labels = mnist_data['train_labels']
            mnist_test_labels = mnist_data['test_labels']
    except FileNotFoundError:
        mnist.datasets_url = url
        # in case the webserver is down, you can also provide mnist data locally
        # mnist.temporary_dir = lambda: os.path.join(os.path.dirname(__file__), 'mnist')

        print('loading and processing mnist for the first time...')

        mnist_train_data = train_images()
        mnist_test_data = test_images()
        mnist_train_labels = train_labels()
        mnist_test_labels = test_labels()

        # plt.figure()
        # plt.imshow(np.reshape(mnist_train_data[0], (28, 28)))
        # plt.show()

        if crop > 0:
            mnist_train_data = mnist_train_data[:, crop:(28 - crop), crop:(28 - crop)]
            mnist_test_data = mnist_test_data[:, crop:(28 - crop), crop:(28 - crop)]

        print('shape before rescaling', mnist_train_data.shape)

        mnist_train_data = rescale(mnist_train_data, sidelen)
        mnist_test_data = rescale(mnist_test_data, sidelen)
        print('shape after rescaling', mnist_train_data.shape)

        mnist_train_data = mnist_train_data / 255.
        mnist_test_data = mnist_test_data / 255.

        print('saving scaled and cropped dataset for future use...')
        np.savez_compressed('./' + dataset_name + str(sidelen) + '_' + str(crop), train_data=mnist_train_data,
                            test_data=mnist_test_data,
                            train_labels=mnist_train_labels, test_labels=mnist_test_labels)

    if shuffle:
        perm = np.random.permutation(len(mnist_train_data))
        mnist_train_data = mnist_train_data[perm]
        mnist_train_labels = mnist_train_labels[perm]

    return mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels


def load_fmnist(sidelen=28, crop=0, shuffle=True):
    return load_mnist(sidelen=sidelen, crop=crop, shuffle=shuffle,
                      url="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
                      dataset_name='fmnist')


if __name__ == '__main__':
    sidelen = 10
    mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=sidelen, crop=4)
    example_im_ori = np.reshape(mnist_train_data[0, :] * -1 + 256, (sidelen, sidelen))
    plt.imshow(example_im_ori, cmap='gray')
    plt.show()
