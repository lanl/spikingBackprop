# coding: utf-8
"""
This code is based on the following paper:
Renner, A., Sheldon, F., Zlotnik, A., Tao, L., & Sornborger, A. T. The Backpropagation Algorithm Implemented
on Spiking Neuromorphic Hardware. arXiv:2106.07030 (2021).

This is a numpy implementation of the backpropagation algorithm for the binary neural network trained on MNIST.
The code is based on Fig. 2 of Renner et al. (2021).
The code is not optimized for speed. (There is also a Tensorflow version).

We don't use a synfire-gated synfirechain (SFC) here, because it is not necessary for the simulation.
Also, some steps are simplified, because in the simulation, we don't need memory units and we don't need to
separate positive and negative activations and weights.

"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from SFC_backprop.input_data import generate_input_data
from SFC_backprop.weight_init import weight_init

loihi_weight_max = 254
loihi_weight_exp = 0

dataset = 'MNIST20'  # To speed up the simulation and for debugging, choose 'MNIST10',
# but the accuracy will be lower (around 90%)
dataset = 'FMNIST28'

# input_threshold = 0.5
# 55000/60000 - acc: 0.7344 - loss: 0.03720744148829766
# train_acc:  0.7318833333333333
# input data:  FMNIST28_test
# val_acc: 0.7343
# val_acc_argmax: 0.7686
# val_acc_argmax_out: 0.7985

# input_threshold = 0.4
# 55000/60000 - acc: 0.6472 - loss: 0.04642928585717144
# train_acc:  0.63895
# input data:  FMNIST28_test
# val_acc: 0.6436
# val_acc_argmax: 0.6726
# val_acc_argmax_out: 0.6998

input_threshold = 0.6
# 55000/60000 - acc: 0.7304 - loss: 0.03650730146029206
# train_acc:  0.7303166666666666
# input data:  FMNIST28_test
# val_acc: 0.7146
# val_acc_argmax: 0.7582
# val_acc_argmax_out: 0.7901


if dataset == 'MNIST20':
    num_pix = 20
    num_in = num_pix ** 2
elif dataset == 'MNIST10':
    num_pix = 10
    num_in = num_pix ** 2
elif dataset == 'MNIST28':
    num_pix = 28
    num_in = num_pix ** 2
elif dataset == 'FMNIST28':
    num_pix = 28
    num_in = num_pix ** 2
else:
    raise Exception('dataset not defined')

num_populations = {
    'in': num_in,
    'hid': 400,
    'out': 10,
    'gat': 1
}

shuffle = True

params = {}
params['sfc_threshold'] = 1024
params['num_populations'] = num_populations
params['weight_exponent'] = loihi_weight_exp

binary_threshold = params['sfc_threshold'] // 2  # sfc_threshold is the Loihi neuron threshold,
# Note that neurons on Loihi are brought half-way to the threshold by the gating, so here, we only have 1/2 * Thr.


def activation_function(x, threshold=binary_threshold):
    return 1 * (x > threshold)


seed = 43
np.random.seed(seed)

num_epochs = 15  # 6
num_trials = 60000

num_plots = 10

input_all, target_all = generate_input_data(num_trials, input_data=dataset, add_bias=False, threshold=input_threshold)

weight_max = loihi_weight_max * (2 ** loihi_weight_exp)
weight_min = -weight_max

np.random.seed(seed + 1)
init_weight_matrix0_ori, init_weight_matrix1_ori = weight_init(params, mode='rand_He')  # 'restore', 'tf'

init_weight_matrix0 = init_weight_matrix0_ori.copy().T
init_weight_matrix1 = init_weight_matrix1_ori.copy().T

mu = 2

W1_sim = init_weight_matrix0
W2_sim = init_weight_matrix1

log_val_acc = []

for ep in range(num_epochs):

    print('Epoch', str(ep + 1) + '/' + str(num_epochs))
    correct = 0
    log_correct = []
    log_W2 = []
    log_W1 = []

    log_mse = []

    if shuffle:
        np.random.seed(seed + ep)
        np.random.shuffle(input_all)
        np.random.seed(seed + ep)
        np.random.shuffle(target_all)

    for i in range(num_trials):
        if 0 == (i % 5000) and i != 0:
            acc5000 = (log_correct[-1] - log_correct[-5000]) / 5000
            print(str(i) + '/' + str(num_trials), '- acc:', acc5000, '- loss:', np.mean(log_mse[-5000:-1]))

        # feedforward path

        # Step 1
        inp = input_all[i] * 1

        # Step 2
        hid = np.dot(inp, W1_sim)

        hid_r = activation_function(hid)
        hid_p = activation_function(hid - binary_threshold + 1)  # derivative of truncated ReLU is 0 if out>1
        hid_m = activation_function(hid + binary_threshold)  # derivative of truncated ReLU is 0 if out<0

        # Step 3
        target = target_all[i]

        out = np.dot(hid_r, W2_sim)

        out_r = activation_function(out)
        out_p = activation_function(out - binary_threshold + 1)  # derivative of truncated ReLU is 0 if hid>1
        out_m = activation_function(out + binary_threshold)  # derivative of truncated ReLU is 0 if hid<0

        b_h = activation_function(hid_m - hid_p, threshold=0)

        # logging (for accuracy and plotting)
        correct += (out_r == target).all()
        log_correct += [correct]

        # backward path

        # Step 4
        err = (target - out_r)
        log_mse += [np.mean(err ** 2)]  # logging for loss calculation

        # d_2 = (target - out_r) * (out_m - out_p)
        d_2p = activation_function((target - out_r) * out_m, 0)
        d_2m = activation_function((out_r - target) * out_m - out_p, 0)

        # Step 5 & 9
        W2_sim += np.outer(hid_r, d_2p - d_2m) * mu  # weight 2 update
        W2_sim = np.clip(W2_sim, -weight_max, weight_max)

        # Step 6 & 10
        d_1 = np.clip(np.dot(d_2p - d_2m, W2_sim.T) * b_h, -1, 1)  # backprop
        # instead of applying the activation function with 0 threshold twice, to create d_1p and d_1m,
        # we just clip it

        # Step 7 & 11
        W1_sim += np.outer(inp, d_1) * mu  # weight 1 update
        W1_sim = np.clip(W1_sim, -weight_max, weight_max)

    print('train_acc: ', correct / num_trials)

    input_test, target_test = generate_input_data(10000, input_data=dataset + '_test', add_bias=False)

    if num_plots > 0:
        fig, axs = plt.subplots(nrows=num_plots, ncols=2)

    correct = 0
    cr_argmax = 0
    cr_argmax_out = 0
    log_correct = []
    log_correct_argmax = []
    for i in range(10000):

        inp = input_test[i]
        target = target_test[i]
        hid = np.dot(inp, W1_sim)
        hid_r = activation_function(hid)
        out = np.dot(hid_r, W2_sim)
        out_r = activation_function(out)

        if i >= 10000 - num_plots:
            axs[i - 10000 + num_plots, 0].imshow(np.reshape(inp, (num_pix, num_pix)))
            axs[i - 10000 + num_plots, 1].plot(out_r, label='out')
            axs[i - 10000 + num_plots, 1].plot(target, label='tgt')

        correct += (out_r == target).all()
        log_correct += [(out_r == target).all()]

        cr_argmax_out += np.sum(np.argmax(target, axis=0) == np.argmax(out, axis=0))
        cr_argmax += np.sum(np.argmax(target, axis=0) == np.argmax(out_r, axis=0))
        log_correct_argmax += [np.sum(np.argmax(target, axis=0) == np.argmax(out_r, axis=0))]


    print('val_acc:', correct / 10000)
    print('val_acc_argmax:', cr_argmax / 10000)
    print('val_acc_argmax_out:', cr_argmax_out / 10000)

    log_val_acc.append(correct / 10000)

    if num_plots > 0:
        plt.legend()
        plt.show()

do_save_weights = False

if do_save_weights:
    str_time = time.strftime("%Y%m%d_%H%M")
    filename = os.path.join(".", "saved_weights", "final_weights_np_" + str_time + ".npz")
    np.savez(filename, W1=W1_sim.T, W2=W2_sim.T, W2Tp=W2_sim, W2Tm=W2_sim, W1_copy=W1_sim.T)

plt.figure()
plt.hist(W1_sim.flatten(), bins=50)
plt.figure()
plt.hist(W2_sim.flatten())
plt.show()

plt.figure()
plt.plot([0.5] + log_val_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.tight_layout()
plt.show()
