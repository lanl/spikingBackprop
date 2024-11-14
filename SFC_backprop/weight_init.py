#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import os
import warnings

import numpy as np

import SFC_backprop


def restore_weights(datadir=os.path.join(".", "saved_weights"), index=0):
    files = os.listdir(datadir)
    files = [f for f in files if 'final_weights' in f]
    files = np.flip(np.sort(files))
    if isinstance(index, str):
        filename = index
    else:
        filename = files[index]
    print('restoring weights from', os.path.join(datadir, filename))
    with np.load(os.path.join(datadir, filename), allow_pickle=True) as f:
        w_final_loaded = {k: f[k] for k in f}
        # w1_r, w2_r, w2Tp_r, w2Tm_r = f['W1'], f['W2'], f['W2Tp'], f['W2Tm']
    return w_final_loaded


def weight_init(params, mode='rand', file=None):
    """
    mode can be rand 'restore', 'tf'
    """

    num_populations = params['num_populations']

    if mode == 'rand':
        # This works for MNIST:
        init_weight_matrix0 = (np.random.randn(num_populations['in'], num_populations['hid']) + 0.5) / 20  # 30
        init_weight_matrix1 = (np.random.randn(num_populations['hid'], num_populations['out']) + 0.5) / 50  # 100#30
    elif mode == 'rand_He':
        # He:
        init_weight_matrix0 = np.random.randn(num_populations['in'], num_populations['hid']) * \
                              np.sqrt(2 / (num_populations['in'] + num_populations['hid']))
        init_weight_matrix1 = np.random.randn(num_populations['hid'], num_populations['out']) * \
                              np.sqrt(2 / (num_populations['hid'] + num_populations['out']))
        # plt.figure()
        # plt.hist(np.ndarray.flatten(init_weight_ma(params['num_populations']trix0), bins=50)
        # plt.hist(np.ndarray.flatten(init_weight_matrix1), bins=50)

    elif mode == 'rand_uniform':
        init_weight_matrix0 = (np.random.rand(num_populations['in'], num_populations['hid']) - 0.5) * 2
        init_weight_matrix1 = (np.random.rand(num_populations['hid'], num_populations['out']) - 0.5) * 2

    elif mode == 'restore':
        print("RESTORING WEIGHTS FROM FILE")
        if file is None:
            # W1_r, W2_r, W2Tp_r, W2Tm_r = restore_weights(datadir=os.path.join(".", "saved_weights"), index=0)
            w_final_loaded = restore_weights(datadir=os.path.join(".", "saved_weights"), index=0)
        else:
            w_final_loaded = restore_weights(datadir=os.path.join(".", "saved_weights"), index=file)
        # plt.figure()
        # plt.hist(np.ndarray.flatten(init_weight_matrix0))
        # plt.hist(np.ndarray.flatten(W1_r))
        # plt.figure()
        # plt.hist(np.ndarray.flatten(init_weight_matrix1), bins=100)
        # init_weight_matrix0 = W1_r.T
        # init_weight_matrix1 = W2_r.T
        init_weight_matrix0 = w_final_loaded['w1']
        init_weight_matrix1 = w_final_loaded['w2']
        # note that in case of restore, the matrices are transposed!

    elif mode == 'tf':
        print("RESTORING WEIGHTS FROM TF FILE")
        with np.load(os.path.join(os.path.dirname(SFC_backprop.__file__), '..', 'MNIST_binary', 'trained_weights',
                                  'weights_mnist.npz'),
                     allow_pickle=True) as f:
            tf_weights0 = f['weights0']
            tf_weights1 = f['weights1']

        init_weight_matrix0 = tf_weights0
        init_weight_matrix1 = tf_weights1
    else:
        raise NotImplementedError

    if not (mode == 'restore' or mode == 'tf'):
        init_weight_matrix0 = weights2loihi(weight_matrix=init_weight_matrix0,
                                            weight_factor=params['sfc_threshold'],
                                            weight_exponent=params['weight_exponent'], weight_lim=(-240, 240))
        init_weight_matrix1 = weights2loihi(weight_matrix=init_weight_matrix1,
                                            weight_factor=params['sfc_threshold'],
                                            weight_exponent=params['weight_exponent'], weight_lim=(-240, 240))
        # the weight limits (just for init) are set a bit below w is possible (254) as it is better if the weights
        # never reach the max due to the asymmetry that can arise as the negative weight can go lower (-256) than
        # the positive (+254)

    return init_weight_matrix0, init_weight_matrix1

    # # Just to test if the loaded weights make sense
    # num_plots = 10
    # fig, axs = plt.subplots(nrows=num_plots, ncols=2)
    # for i in range(num_plots):
    #     input = input_all[30 + 5 * i][[0, 2, 4, 6, 8, 10, 12, 14, 16, 18] + list(range(20, 110))]
    #     hid = np.dot(input,init_weight_matrix0)
    #     hid_r = np.clip(hid,0,1)
    #     hid_r = np.round(hid_r)
    #     out = np.dot(hid_r,init_weight_matrix1)
    #     out_r = np.clip(out,0,1)
    #     out_r = np.round(out_r)
    #
    #     axs[i, 0].imshow(np.reshape(input, (10, 10)))
    #     axs[i, 1].plot(out_r)


def weights2loihi(weight_matrix, weight_factor, weight_exponent=0, weight_lim=(-254, 254)):
    weight_min = weight_lim[0]  # Should be the same
    weight_max = weight_lim[1]

    weight_factor_exp = weight_factor * 2 ** -weight_exponent

    weight_matrix_loihi = weight_matrix.copy()

    weight_matrix_loihi *= weight_factor_exp

    # np.max(weight_matrix_loihi * weight_factor_exp)
    # np.min(weight_matrix_loihi * weight_factor_exp)
    weight_matrix_loihi[np.where(weight_matrix_loihi < weight_min)] = weight_min
    weight_matrix_loihi[np.where(weight_matrix_loihi > weight_max)] = weight_max

    weight_matrix_loihi = np.floor(np.abs(weight_matrix_loihi)/2) * 2 * np.sign(weight_matrix_loihi)

    print('w max: ', np.max(weight_matrix_loihi))

    if np.sum(weight_matrix * weight_factor_exp != weight_matrix_loihi) != 0:
        warnings.warn("rounding of init matrix")

    return weight_matrix_loihi.T
