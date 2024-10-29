#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by : Alpha Renner (alpren@ini.uzh.ch)

"""
import os
import sys
import warnings
import argparse

import matplotlib.pyplot as plt
import numpy as np

try:
    print(sys.path)
    print(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except:
    pass
from SFC_backprop.network_parameters_2layer_MNIST import params
from SFC_backprop.backprop_network import BackpropNet

# to reset Kapohobay run something like:
# lsusb
# sudo usbreset 04b4:6572

parser = argparse.ArgumentParser()
parser.add_argument("--weight_mode", default="rand_He", help="Set the weight mode, either rand_He to initialize randomly or restore to restore from the most recently saved weights")
args = parser.parse_args()

print(f"Weight mode: {args.weight_mode}")


try:
    os.environ['SLURM']
    params['on_kapohobay'] = False
    os.environ['PARTITION'] = "nahuku32_2h"
    os.environ['BOARD'] = "ncl-ext-ghrd-01"
except:
    print('SLURM not set. Running on Kapohobay!')
    os.environ['KAPOHOBAY'] = '1'
    params['on_kapohobay'] = True

weight_file = None
weight_mode = args.weight_mode 

do_probe_energy = False
do_train = True
do_plots = False
do_debug = False  
dataset = 'MNIST20'
# dataset = 'MNIST10'
# dataset = 'MNIST28'
# dataset = 'FMNIST28'

if do_train:
    params['dataset'] = dataset
    probe_mode = 1
else:
    params['num_trials'] = 10000
    # params['num_trials'] = 1000
    params['dataset'] = dataset + '_test'
    probe_mode = 3

if do_probe_energy:
    probe_mode = 0
    params['num_trials'] = 10000

if do_debug:
    probe_mode = 2
    seed = 42
    params['num_trials'] = 300
else:
    seed = np.random.randint(0, 10000)  # the seed needs to be different in each epoch

np.random.seed(seed)
print('seed:', seed)

params['seed'] = seed
params['do_train'] = do_train
params['do_probe_energy'] = do_probe_energy
params['weight_mode'] = weight_mode
params['weight_file'] = weight_file

bp_sfc = BackpropNet(params, debug=0)
bp_sfc.setup_probes(probe_mode=probe_mode)

bp_sfc.run()
if not do_probe_energy:  # when energy is probed, all other probes are disabled
    bp_sfc.save_results()
    bp_sfc.accuracy_from_weights()

# input_data, output_data = generate_input_data(10000, input_data=dataset, add_bias=False)
# validate_inference_activity(bp_sfc, labels=bp_sfc.output_data, do_plots=False)

# bp_sfc.load_spikes('./saved_spikes/spikes_20210405_1611.npz')

spikes = bp_sfc.spikes
w_final = bp_sfc.w_final

try:
    # sparsity
    for lay in spikes:
        print(lay,
              np.sum(spikes[lay][:, bp_sfc.num_gate:]) / (spikes[lay][:, bp_sfc.num_gate:].shape[1] // bp_sfc.num_gate))
except TypeError:
    pass

if do_plots:
    # fig, ax = plt.subplots(len(bp_sfc.spikeprobes), 1, sharex=True)
    # for i, lay in enumerate(bp_sfc.spikeprobes):
    #     if 'g' in lay:
    #         inds, times = np.where(spikes[lay])
    #         ax[i].plot(times, inds, '.')
    #         ax[i].set_ylabel(lay)
    # tick_ts = np.arange(1, 100 * bp_sfc.bp_sfc.num_gate, 2)  # bp_sfc.bp_sfc.num_gate)
    # _ = plt.xticks(tick_ts, tick_ts % bp_sfc.bp_sfc.num_gate)

    plt.figure()
    for i, lay in enumerate(['o', 't']):
        inds, times = np.where(spikes[lay])
        plt.plot(times, inds + 0.1 * i, '.', label=lay)
    plt.legend()
    tick_ts = np.arange(1, 100 * bp_sfc.num_gate, 2)  # bp_sfc.bp_sfc.num_gate)
    _ = plt.xticks(tick_ts, tick_ts % bp_sfc.num_gate)

    ## Annotated spike plot
    leg_loc = 'upper center'
    leg_col = 'lightgrey'
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12.8, 9.6), gridspec_kw={'height_ratios': [1, 2, 2, 1]})
    ax[0].set_ylabel('out\nneuron index')
    ax[1].set_ylabel('hid\nneuron index')
    ax[2].set_ylabel('in\nneuron index')
    ax[3].set_ylabel('gate\nneuron index')

    inds, times = np.where(spikes['o'])
    ax[0].plot(times, inds, '.', label='$o$')
    try:
        inds, times = np.where(spikes['o_copy'])
        ax[0].plot(times + 0.15, inds, '.', label='$o^<$')
        inds, times = np.where(spikes['o_copy2'])
        ax[0].plot(times + 0.3, inds, '.', label='$o^>$')
        inds, times = np.where(spikes['oT-'])
        ax[0].plot(times + 0.15, inds, '.', label='$o^{T-}$')
        inds, times = np.where(spikes['t'])
        ax[0].plot(times + 0.15, inds, '.', label='$t$')
        inds, times = np.where(spikes['d-'])
        ax[0].plot(times, inds, '.', label='$d_2^-$')
        inds, times = np.where(spikes['d+'])
        ax[0].plot(times, inds, '.', label='$d_2^+$')
    except:
        pass
    ax[0].legend(ncol=2, loc=leg_loc, facecolor=leg_col, framealpha=1)

    inds, times = np.where(spikes['h1'])
    ax[1].plot(times, inds, '.', label='$h$')
    try:
        inds, times = np.where(spikes['h1T'])
        ax[1].plot(times + 0.15, inds, '.', label='$d_1$')
        inds, times = np.where(spikes['h1_copy'])
        ax[1].plot(times + 0.15, inds, '.', label='$h^<$')
        inds, times = np.where(spikes['h1_copy2'])
        ax[1].plot(times + 0.3, inds, '.', label='$h^>$')
        # inds, times = np.where(spikes['m_h1'])
        # ax[1].plot(times, inds+0.1, '.', label='m_h1')
    except:
        pass
    ax[1].legend(ncol=2, loc=leg_loc, facecolor=leg_col, framealpha=1)

    inds, times = np.where(spikes['x'])
    ax[2].plot(times, inds, '.', label='$x$')
    ax[2].legend(loc=leg_loc, facecolor=leg_col, framealpha=1)

    tick_ts = np.arange(1, 100 * bp_sfc.num_gate, 1)  # bp_sfc.bp_sfc.num_gate)
    _ = plt.xticks(tick_ts, tick_ts % bp_sfc.num_gate)

    ax[3].plot(tick_ts, (tick_ts + 1) % bp_sfc.num_gate, '.', label='g1-11')
    ax[3].legend(loc=leg_loc, facecolor=leg_col, framealpha=1)
    ax[3].set_xlabel('timestep')
    import string

    for n, a in enumerate(ax):
        a.text(-0.05, 1.05, string.ascii_uppercase[n], transform=a.transAxes,
               size=18, weight='bold')

    # Annotations are for spikes spikes_20210405_1611.npz
    plt.xlim(14 * bp_sfc.num_gate, 20 * bp_sfc.num_gate)

    ax[0].text(0.075, 0.15, '(1)', transform=ax[0].transAxes, size=13)
    ax[0].text(0.135, 0.15, '(2)', transform=ax[0].transAxes, size=13)
    ax[0].text(0.215, 0.25, '(3)', transform=ax[0].transAxes, size=13)
    ax[0].text(0.38, 0.8, '(4)', transform=ax[0].transAxes, size=13)
    ax[0].text(1 - 0.13, 0.23, '(5)', transform=ax[0].transAxes, size=13)
    ax[0].text(1 - 0.05, 0.23, '(6)', transform=ax[0].transAxes, size=13)

    ax[1].text(0.08, 0.5, '(7)', transform=ax[1].transAxes, size=13)
    ax[1].text(0.14, 0.5, '(8)', transform=ax[1].transAxes, size=13)

    ax[0].set_ylim((0, 10))
    ax[1].set_ylim((0, 50))
    ax[2].set_ylim((0, 50))
    # fig.legend()
    plt.tight_layout()
    plt.tight_layout()

    import matplotlib

    matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.savefig('./rasterplot_annotated.svg')

if do_train and not do_probe_energy:
    try:
        print(np.sum(bp_sfc.get_activity('h1', 2)))
        print(np.sum(bp_sfc.get_activity('o', 3)))
        for i in [0, 1, 3, 4, 6, 8, 10]:
            print('phase', i, end=': ')
            if not 0 == (np.sum(bp_sfc.get_activity('h1', i))):
                warnings.warn('h1 not silent in inactive phase ' + str(i))
            if not 0 == (np.sum(bp_sfc.get_activity('h1_copy', i))):
                warnings.warn('h1_copy not silent in inactive phase ' + str(i))
            if not 0 == (np.sum(bp_sfc.get_activity('h1_copy2', i))):
                warnings.warn('h1_copy2 not silent in inactive phase ' + str(i))
        print('')

        for i in [0, 1, 2, 4, 6, 7, 8, 10, 11]:
            print('phase', i, end=': ')
            if not 0 == (np.sum(bp_sfc.get_activity('o', i))):
                warnings.warn('o not silent in inactive phase ' + str(i))
            if not 0 == (np.sum(bp_sfc.get_activity('o_copy', i))):
                warnings.warn('o_copy not silent in inactive phase ' + str(i))
            if not 0 == (np.sum(bp_sfc.get_activity('o_copy2', i))):
                warnings.warn('o_copy2 not silent in inactive phase ' + str(i))
            if not 0 == (np.sum(bp_sfc.get_activity('o', i))):
                warnings.warn('o not silent in inactive phase ' + str(i))
        print('')

        for i in [0, 1, 2, 3, 4, 7, 8, 11]:
            if not 0 == (np.sum(bp_sfc.get_activity('h1T', i))):
                warnings.warn('h1T not silent in inactive phase ' + str(i))

        print('x:')
        for i in range(bp_sfc.num_gate):
            print('phase', i, end=': ')
            print(np.sum(bp_sfc.get_activity('x', i)))

        print('h1:')
        for i in range(bp_sfc.num_gate):
            print('phase', i, end=': ')
            print(np.sum(bp_sfc.get_activity('h1', i)))

        print('d:')
        for i in range(bp_sfc.num_gate):
            print('phase', i, end=': ')
            print(np.sum(bp_sfc.get_activity('d-', i)))
            print(np.sum(bp_sfc.get_activity('d+', i)))

        print('o:')
        for i in range(bp_sfc.num_gate):
            print('phase', i, end=': ')
            print(np.sum(bp_sfc.get_activity('o', i)), end=': ')
            print(np.sum(bp_sfc.get_activity('o_copy', i)), end=': ')
            print(np.sum(bp_sfc.get_activity('o_copy2', i)), end=': ')
            print(np.sum(bp_sfc.get_activity('oT-', i)))

    except KeyError:
        print('not checked for errors. Run in debug mode with all probes to check for errors.')

    # bp_sfc.get_activity('x', 1)[:100]==bp_sfc.get_activity('x', 11)
    try:
        if not (w_final['w1'] == w_final['w1_copy1']).all():
            warnings.warn('w1 not equal to w1_copy1')
        if not (w_final['w1'] == w_final['w1_copy2']).all():
            warnings.warn('w1 not equal to w1_copy2')
    except KeyError as e:
        print(e)

    try:
        if not (w_final['w1_copy1'] == w_final['w1_copy2']).all():
            warnings.warn('w1_copy1 not equal to w1_copy2')
        if not (w_final['w2'] == w_final['w2_copy1']).all():
            warnings.warn('w2 not equal to w2_copy1')
        if not (w_final['w2'] == w_final['w2_copy2']).all():
            warnings.warn('w2 not equal to w2_copy2')
    except KeyError as e:
        print(e)

    if not (w_final['w2'].T == w_final['w2Tp']).all():
        warnings.warn('w2 not equal to w2Tp')
    print(np.sum(np.abs(w_final['w2'].T - w_final['w2Tp'])))

    if not (w_final['w2'].T == -w_final['w2Tm']).all():
        warnings.warn('w2 not equal to -w2Tm')
    print(np.sum(np.abs(w_final['w2'].T + w_final['w2Tm'])))

    if not (w_final['w2Tp'] == -w_final['w2Tm']).all():
        warnings.warn('w2Tp not equal to -w2Tm')
    print(np.sum(np.abs(w_final['w2Tm'] + w_final['w2Tp'])))

    print(w_final['w2Tm'][np.where(w_final['w2'].T != -w_final['w2Tm'])])
    print(w_final['w2'].T[np.where(w_final['w2'].T != -w_final['w2Tm'])])

    print(w_final['w2Tp'][np.where(w_final['w2'].T != w_final['w2Tp'])])
    print(w_final['w2'].T[np.where(w_final['w2'].T != w_final['w2Tp'])])
