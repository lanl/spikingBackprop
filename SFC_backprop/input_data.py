#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from SFC_backprop.load_mnist import load_mnist


def relu(x):
    ret = np.asarray(np.maximum(np.asarray(x), 0))
    return ret


def generate_input_data(num_trials, input_data='MNIST10', add_bias=False):
    print('input data: ', input_data)

    if input_data == 'MNIST10':
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=10, crop=4)
        if num_trials > 60000:
            for i in range(int(np.ceil((num_trials - 60000) / 60000))):
                mnist_train_data2, mnist_test_data2, mnist_train_labels2, mnist_test_labels2 = load_mnist(sidelen=10,
                                                                                                          crop=4)
                mnist_train_data = np.concatenate((mnist_train_data, mnist_train_data2), axis=0)
                mnist_train_labels = np.concatenate((mnist_train_labels, mnist_train_labels2))

        train_input = np.round(mnist_train_data[0:num_trials] * 256, 2)

        train_output = np.zeros((num_trials, 10))
        for i, l in enumerate(mnist_train_labels[0:num_trials]):
            train_output[i, l] = 1
    elif input_data == 'MNIST10_test':
        assert num_trials <= 10000, 'test set only has 10000 samples'
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=10, crop=4)
        train_input = np.round(mnist_test_data[0:num_trials] * 256, 2)
        train_output = np.zeros((num_trials, 10))
        for i, l in enumerate(mnist_test_labels[0:num_trials]):
            train_output[i, l] = 1
    elif input_data == 'MNIST20':
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=20, crop=4)
        if num_trials > 60000:
            for i in range(int(np.ceil((num_trials - 60000) / 60000))):
                mnist_train_data2, mnist_test_data2, mnist_train_labels2, mnist_test_labels2 = load_mnist(sidelen=20,
                                                                                                          crop=4)
                mnist_train_data = np.concatenate((mnist_train_data, mnist_train_data2), axis=0)
                mnist_train_labels = np.concatenate((mnist_train_labels, mnist_train_labels2))

        train_input = np.round(mnist_train_data[0:num_trials] * 256, 2)
        train_output = np.zeros((num_trials, 10))
        for i, l in enumerate(mnist_train_labels[0:num_trials]):
            train_output[i, l] = 1
    elif input_data == 'MNIST20_test':
        assert num_trials <= 10000, 'test set only has 10000 samples'
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = load_mnist(sidelen=20, crop=4)
        train_input = np.round(mnist_test_data[0:num_trials] * 256, 2)
        train_output = np.zeros((num_trials, 10))
        for i, l in enumerate(mnist_test_labels[0:num_trials]):
            train_output[i, l] = 1
    else:
        print('input_data not recognized')
        raise ValueError('input_data not recognized')

    if add_bias:
        train_input = np.concatenate((train_input, np.ones(shape=(num_trials, 1))), axis=1)

    train_input = (0.5 < train_input)
    # train_input = (np.reshape(train_input.mean(axis=1), (len(train_input), 1)) < train_input)

    return train_input, train_output
