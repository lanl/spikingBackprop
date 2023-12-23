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


def validate_inference_activity(bp_sfc, labels=None, inp=None, do_plots=True):
    """
    validate the activities based on current weight
    """

    binary_threshold = bp_sfc.params['sfc_threshold'] // 2

    def round(x):
        x_r = 1 * (x > binary_threshold)
        return x_r

    num_in = bp_sfc.params['num_populations']['in']
    num_hid = bp_sfc.params['num_populations']['hid']
    num_out = bp_sfc.params['num_populations']['out']

    try:
        in_vec = bp_sfc.get_activity(in_name, phase_in)
    except KeyError as e:
        print(e)
        if inp is not None:
            in_vec = inp
        else:
            raise Exception('no input spikes provided')

    try:
        hid_vec = bp_sfc.get_activity(hid_name, phase_hid)
    except KeyError as e:
        print(e)

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

        hid_calc = round(np.dot(in_vec, W1.T[np.newaxis, :, :])[:, 0, :])
        out_calc_calc = round(np.dot(hid_calc, W2.T[np.newaxis, :, :])[:, 0, :])

        try:
            out_calc = round(np.dot(hid_vec, W2.T[np.newaxis, :, :])[:, 0, :])
        except:
            pass

        try:
            if do_plots:
                fig, axs = plt.subplots(num_out, 1)
                if num_out == 1:
                    axs = [axs]
                for i, ax in enumerate(axs):
                    ax.plot(out_vec[:, i])
                    ax.plot(out_calc[:, i])
                    # ax.plot(out_calc_calc[:, i])
                    ax.plot(tgt_vec[:, i])
                plt.legend(['out',  # 'calc',
                            'calc',  # 'calc_calc',
                            'tgt'])

                fig, axs = plt.subplots(num_out, 1)
                if num_out == 1:
                    axs = [axs]
                for i, ax in enumerate(axs):
                    norm_out_calc_calc = out_calc_calc[:, i] - np.min(out_calc_calc[:, i])
                    norm_out_calc_calc = norm_out_calc_calc / np.max(norm_out_calc_calc)
                    ax.plot(0.4 < norm_out_calc_calc)
                    ax.plot(tgt_vec[:, i])
                    # print('classification calc_calc', i, ': ',   np.sum((0.3 < norm_out_calc_calc) == tgt_vec[:, i]))
                    print('correct output', i, ': ', np.sum(out_vec[:, i] == tgt_vec[:, i]))
                plt.legend(['calc_calc_round', 'tgt'])

        except NameError:
            pass

    try:
        cr_calc_calc = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out_calc_calc, axis=1)) / len(tgt_vec)
        print('calc_calc MNIST classification rate (out vs. target activity): ', cr_calc_calc)
        correct = 0
        for i in range(len(out_calc_calc)):
            if (out_calc_calc[i] == tgt_vec[i]).all():
                correct += 1
        print('actual fully correct', correct / len(out_calc_calc))

        err = out_calc_calc - tgt_vec
        print('loss mse calc:', np.mean(err ** 2))
    except NameError:
        pass

    try:
        print('same as calc: ', np.sum(out_calc == out_vec) / np.prod(out_calc.shape))
        cr_calc = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out_calc, axis=1)) / len(tgt_vec)
        print('calc MNIST classification rate (out vs. target activity): ', cr_calc)
    except:
        pass

    try:
        cr = np.sum(np.argmax(tgt_vec, axis=1) == np.argmax(out_vec, axis=1)) / len(tgt_vec)
        print('actual MNIST classification rate (out vs. target activity): ', cr)
    except NameError:
        pass

    if do_plots:
        for i in range(len(out_vec)):
            if not (out_vec[i] == (out_calc[i] > bp_sfc.params['sfc_threshold']) * 1).all():
                print(i)
                print(out_calc[i])
                print(out_vec[i])

        correct = 0
        for i in range(len(out_vec)):
            if (tgt_vec[i] == (out_calc[i] > bp_sfc.params['sfc_threshold']) * 1).all():
                correct += 1
        print('actual fully correct calc', correct / len(out_vec))

    try:
        correct = 0
        for i in range(len(out_vec)):
            if (out_vec[i] == tgt_vec[i]).all():
                correct += 1
        print('actual fully correct', correct / len(out_vec))
    except NameError:
        pass

    if do_plots:
        num_plots = 10
        img_width = int(np.sqrt(in_vec.shape[1]))
        fig, axs = plt.subplots(nrows=num_plots, ncols=2)
        for i in range(num_plots):
            axs[i, 0].imshow(np.reshape(in_vec[i + 1], (img_width, img_width)))
            # axs[i, 1].plot(out_round[i+1], label='round')
            axs[i, 1].plot(out_vec[i + 1], label='out')
            axs[i, 1].plot(tgt_vec[i + 1], label='tgt')
        plt.legend()

        num_plots = 10
        fig, axs = plt.subplots(nrows=num_plots, ncols=2)
        for i in range(num_plots):
            axs[i, 0].imshow(np.reshape(in_vec[-i], (img_width, img_width)))
            axs[i, 1].plot(out_vec[-i], label='out')
            # axs[i, 1].plot(tgt_vec[i+1], label='tgt')
        plt.legend()

        num_plots = 10
        fig, axs = plt.subplots(nrows=num_plots, ncols=1)
        for i in range(num_plots):
            axs[i].plot(hid_vec[i + 1], label='out')
            # axs[i, 1].plot(tgt_vec[i+1], label='tgt')
        plt.legend()
