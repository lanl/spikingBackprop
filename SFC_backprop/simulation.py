#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np

NUM_PHASES = 1

in_name = 'x'
hid_name = 'h1'
out_name = 'o'
hid_T_p = 'h1T+'
out_T_p = 'oT+'
hid_T_m = 'h1T-'
out_T_m = 'oT-'

phase_in = 1
phase_hid = 2
phase_out = 3
phase_err = 4
phase_W2pot = 5
phase_W2dep = 9
phase_W1pot = 7
phase_W1dep = 11


def inference_from_weights(bp_sfc, labels=None, inp=None):
    """
    validate the activities based on current weight
    """

    binary_threshold = bp_sfc.params['sfc_threshold'] // 2

    def round(x):
        x_r = 1 * (x > binary_threshold)
        return x_r

    num_out = bp_sfc.params['num_populations']['out']

    if inp is not None:
        in_vec = inp
    else:
        try:
            in_vec = bp_sfc.get_activity(in_name, phase_in)
        except Exception as e:
            print(e)
            raise Exception('no input spikes provided')

    try:
        out_vec = bp_sfc.get_activity(out_name, phase_out)
    except KeyError:
        print('no output spikes provided, calculate from weights')

    if labels is None:
        tgt_vec = bp_sfc.get_activity('t', phase_out)
    else:
        labels = np.asarray(labels, dtype=int)
        tgt_vec = labels

    w_final = bp_sfc.w_final
    if w_final:
        W1 = w_final['w1']
        W2 = w_final['w2']

        hid = round(np.dot(in_vec, W1.T[np.newaxis, :, :])[:, 0, :] * 2**bp_sfc.params['weight_exponent'])
        out_v = np.dot(hid, W2.T[np.newaxis, :, :])[:, 0, :] * 2**bp_sfc.params['weight_exponent']
        out = round(out_v)

        cr_calc = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out, axis=1)) / len(tgt_vec)
        cr_calc_v = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out_v, axis=1)) / len(tgt_vec)
        print('calculated MNIST classification rate (out vs. target activity): ', cr_calc)
        print('calculated MNIST classification rate from activity: ', cr_calc_v)
        correct = 0
        for i in range(len(out)):
            if (out[i] == tgt_vec[i]).all():
                correct += 1
        print('exactly correct', correct / len(out))

        err = out - tgt_vec
        print('loss mse:', np.mean(err ** 2))

    try:
        print('same as calc: ', np.sum(out == out_vec) / np.prod(out.shape))
    except:
        pass

    try:
        cr = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out_vec, axis=1)) / len(tgt_vec)
        print('chip MNIST classification rate (out vs. target activity): ', cr)

        correct = 0
        for i in range(len(out_vec)):
            if (out_vec[i] == tgt_vec[i]).all():
                correct += 1
        print('exactly correct', correct / len(out_vec))
    except NameError:
        pass

