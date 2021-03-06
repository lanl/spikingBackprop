#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import mnist
import numpy as np
import skimage


def rescale(data, sidelen=28):
    scaled = []
    for i in np.arange(data.shape[0]):
        s = skimage.transform.resize(data[i], (sidelen, sidelen), anti_aliasing=False, mode='constant')
        scaled.append(s)
    return np.asarray(scaled).reshape(-1, sidelen * sidelen)


def load_mnist(sidelen=28, crop=4, shuffle=True):
    """
    First cropping on all sides, then rescaling to sidelen x sidelen.
    Args:
        sidelen:
        crop:
    """
    try:
        with np.load('./mnist' + str(sidelen) + '_' + str(crop) + '.npz') as mnist_data:
            mnist_train_data = mnist_data['train_data']
            mnist_test_data = mnist_data['test_data']
            mnist_train_labels = mnist_data['train_labels']
            mnist_test_labels = mnist_data['test_labels']
    except FileNotFoundError:
        # mnist.datasets_url = 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/'
        mnist.temporary_dir = lambda: os.path.join(os.path.dirname(__file__), 'mnist')

        mnist_train_data = mnist.train_images()
        mnist_test_data = mnist.test_images()
        mnist_train_labels = mnist.train_labels()
        mnist_test_labels = mnist.test_labels()
        if crop > 0:
            mnist_train_data = mnist_train_data[:, crop:(28 - crop), crop:(28 - crop)]
            mnist_test_data = mnist_test_data[:, crop:(28 - crop), crop:(28 - crop)]

        print('shape before rescaling', mnist_train_data.shape)
        mnist_train_data = rescale(mnist_train_data, sidelen) / 255.
        mnist_test_data = rescale(mnist_test_data, sidelen) / 255.
        print('shape after rescaling', mnist_train_data.shape)

        np.savez_compressed('./mnist' + str(sidelen) + '_' + str(crop), train_data=mnist_train_data,
                            test_data=mnist_test_data,
                            train_labels=mnist_train_labels, test_labels=mnist_test_labels)

    if shuffle:
        perm = np.random.permutation(len(mnist_train_data))
        mnist_train_data = mnist_train_data[perm]
        mnist_train_labels = mnist_train_labels[perm]

    return mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels

if __name__ == '__main__':
    # images = mnist.train_images()
    # example_im = skimage.transform.resize(images[0,:,:] * -1 + 256, 10*np.asarray(images[0,:,:].shape, anti_aliasing=False)
    # plt.figure()
    # plt.imshow(example_im, cmap='gray')

    # example_im_ori = images[0, :, :] * -1 + 256
    # plt.imshow(example_im_ori, cmap='gray')

    sidelen = 10
    mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=sidelen, crop=4)
    example_im_ori = np.reshape(mnist_train_data[0, :] * -1 + 256, (sidelen, sidelen))
    plt.imshow(example_im_ori, cmap='gray')
