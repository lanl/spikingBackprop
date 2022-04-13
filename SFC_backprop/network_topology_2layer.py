#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by : Alpha Renner (alpren@ini.uzh.ch)

Abbreviations:
inp: input
sfc: main (graded) synfirechain network
g: gating SFC
"""

import numpy as np
import re

# source, target, type
# step number is "first appearance"
connected_pairs = [
    # x mem
    ('x', 'm_x6', '1e_d4'),  # x mem

    # h mem
    ('h1', 'm_h1', '1e'),  # h1 mem

    # step 1->2
    ('x', 'h1', 'p'),  # W1 forward
    ('x', 'h1_copy', 'p'),
    ('x', 'h1_copy2', 'p'),

    # step 2->3
    ('h1', 'o', 'p'),  # W2 forward
    ('h1', 'o_copy', 'p'),
    ('h1', 'o_copy2', 'p'),

    ('o_copy', 'd-', 'g1s'),  # m
    ('o_copy', 'd+', 'g1s'),
    ('o_copy2', 'd-', '1i'),  # p
    # ('o_copy2', 'd+', '1i'), # this is not necessary since o is always active when o_copy2 is active,
    # so it's already inhibited

    # step 3->4
    ('o', 'd-', '1e'),  # o-t
    ('t', 'd-', '1i'),  # o-t
    ('t', 'd+', '1e'),  # t-o
    ('o', 'd+', '1i'),  # t-o

    # step 4->5
    ('d-', 'o', '1e_d4'),  # W2 depression phase
    ('d-', 'o_copy', '1e_d4'),  # W2 depression phase
    ('d-', 'o_copy2', '1e_d4'),  # W2 depression phase
    ('d-', 'oT-', '1e'),  # -W2.T potentiation phase

    ('d+', 'o', '1e'),  # W2 potentiation phase
    ('d+', 'o_copy', '1e'),  # W2 potentiation phase
    ('d+', 'o_copy2', '1e'),  # W2 potentiation phase
    ('d+', 'oT-', '1e_d4'),  # -W2.T depression phase

    ('m_h1', 'h1', '1e_d1'),  # W2 potentiation phase
    ('m_h1', 'h1T', '1e_d1'),  # +W2.T potentiation phase

    # step 5->6
    ('o', 'h1T', 'p'),  # +W2.T connection
    ('oT-', 'h1T', 'p'),  # -W2.T connection

    # # step 6 -> 7
    ('h1T', 'h1', '1e'),  # n for W1 potentiation phase
    ('h1T', 'h1_copy', '1e'),  # n for W1 potentiation phase
    ('h1T', 'h1_copy2', '1e'),  # n for W1 potentiation phase
    ('m_x6', 'x', '1e'),  # x for W1 potentiation phase


    # step 8 -> 9
    ('m_h1', 'h1', '1e_d5'),  # o for W2 potentiation phase
    ('m_h1', 'h1T', '1e_d5'),  # o for +W2.T depression phase

    # step 10 -> 11
    ('m_x6', 'x', '1e_d4'),  # x for W1 depression phase

    # gate

    # x mem
    ('g06', 'm_x6', 'g'),

    # h mem
    ('g03', 'm_h1', 'g'),

    # 01 input
    ('g01', 'x', 'g'),

    # 02 hidden
    ('g02', 'h1', 'g'),
    ('g02', 'h1_copy', 'gm05'),
    ('g02', 'h1_copy2', 'gp05'),
    ('g02', 'h1_copy', 'g'),

    # 03 output
    ('g03', 'o', 'g'),
    ('g03', 'o_copy', 'gm05'),
    ('g03', 'o_copy2', 'gp05'),
    ('g03', 'o_copy', 'g'),
    ('g03', 't', 'g'),

    ('g04', 'h1T', 'gi'),

    # 05 W2 pot
    ('g05', 'h1', 'g'),
    ('g05', 'h1T', 'g'),

    ('g05', 'oT-', 'g'),
    ('g05', 'o', 'g'),
    ('g05', 'o_copy', 'g'),
    ('g05', 'o_copy2', 'g'),

    # 06 backprop pot
    ('g06', 'h1T', 'gm05'),
    ('g06', 'h1T', 'g'),

    ('g06', 'h1', 'gi'),
    ('g06', 'h1_copy', 'gi'),
    ('g06', 'h1_copy2', 'gi'),
    ('g06', 'o', 'gi'),
    ('g06', 'o_copy', 'gi'),
    ('g06', 'o_copy2', 'gi'),

    # 07 W1 pot
    ('g07', 'x', 'g'),

    ('g08', 'o', 'gi'),
    ('g08', 'o_copy', 'gi'),
    ('g08', 'o_copy2', 'gi'),
    ('g08', 'h1', 'gi'),
    ('g08', 'h1_copy', 'gi'),
    ('g08', 'h1_copy2', 'gi'),

    # 09 W2 dep
    ('g09', 'h1', 'g'),
    ('g09', 'h1T', 'g'),

    ('g09', 'o', 'g'),
    ('g09', 'o_copy', 'g'),
    ('g09', 'o_copy2', 'g'),
    ('g09', 'oT-', 'g'),

    # 10 backprop dep
    ('g10', 'h1T', 'gm05'),
    ('g10', 'h1T', 'g'),

    ('g10', 'h1', 'gi'),
    ('g10', 'h1_copy', 'gi'),
    ('g10', 'h1_copy2', 'gi'),
    ('g10', 'o', 'gi'),
    ('g10', 'o_copy', 'gi'),
    ('g10', 'o_copy2', 'gi'),

    # 11 W1 dep
    ('g11', 'x', 'g'),

    ('g00', 'o', 'gi'),
    ('g00', 'o_copy', 'gi'),
    ('g00', 'o_copy2', 'gi'),
    ('g00', 'h1', 'gi'),
    ('g00', 'h1_copy', 'gi'),
    ('g00', 'h1_copy2', 'gi'),

    # connected_pairs_cond
    ('c_h1', 'h1', 'g1'),
    ('c_h1', 'h1', 'g1_d4'),
    ('c_h1', 'h1_copy', 'g1'),
    ('c_h1', 'h1_copy', 'g1_d4'),
    ('c_h1', 'h1_copy2', 'g1'),
    ('c_h1', 'h1_copy2', 'g1_d4'),

    # connected_pairs_inp_sfc
    ('input', 'x', '1e'),
    ('in_tgt', 't', '1e'),

    # connected_pairs_sfc_p
    ('h1_copy', 'c_h1', '1e_d3'),  # fm
    ('h1_copy2', 'c_h1', '1i_d3'),  # fp

    # connected_pairs_g_p
    ('g06', 'c_h1', 'g'),

    # connected_pairs_inp_g
    ('in_g', 'g00', '1ge'),

    # connected_pairs_rew
    # ('g05', 'rew', '1ge'),
    # ('g07', 'rew', '1ge'),
]

gat_layers = np.unique([p[0] for p in connected_pairs if re.match("g[0-9][0-9]", p[0])])
num_gat = max([int(l[1:]) for l in gat_layers])+1
connected_pairs += [("g" + str(i).rjust(2, '0'), "g" + str(i + 1).rjust(2, '0'), "1ge") for i in range(num_gat - 1)] + \
                   [("g" + str(num_gat - 1).rjust(2, '0'), "g00", "1ge")]

# Separate input layer in order to allow a different amount of inputs than outputs
layers = {
    # layer_name : ('layer_type','neuron_type')
    'm_x6': ('in', 'n_sfc'),
    'm_h1': ('hid', 'n_sfc'),
    'c_h1': ('hid', 'n_sfc'),
    't': ('out', 'n_sfc'),
    'd-': ('out', 'n_sfc'), 'd+': ('out', 'n_sfc'),

    'h1T': ('hid', 'n_sfl'),
    'oT-': ('out', 'n_sfl'),

    'input': ('in', 'n_inp'), 'in_tgt': ('out', 'n_inp'), 'in_g': ('gat', 'n_inp'),

    'g00': ('gat', 'n_gat'), 'g01': ('gat', 'n_gat'), 'g02': ('gat', 'n_gat'), 'g03': ('gat', 'n_gat'),
    'g04': ('gat', 'n_gat'), 'g05': ('gat', 'n_gat'), 'g06': ('gat', 'n_gat'), 'g07': ('gat', 'n_gat'),
    'g08': ('gat', 'n_gat'), 'g09': ('gat', 'n_gat'), 'g10': ('gat', 'n_gat'), 'g11': ('gat', 'n_gat'),

    # 'rew': ('gat', 'n_rew'),

    'x': ('in', 'n_sfc'),
    'h1': ('hid', 'n_sfl'), 'h1_copy': ('hid', 'n_sfl'), 'h1_copy2': ('hid', 'n_sfl'),
    'o': ('out', 'n_sfl'), 'o_copy': ('out', 'n_sfl'), 'o_copy2': ('out', 'n_sfl'),
}

topology_learn = {'connected_pairs': connected_pairs, 'layers': layers}

connected_pairs_inf = [
    ('x', 'h1', 'f'),  # W1 forward
    ('h1', 'o', 'f'),  # W2 forward

    ('g01', 'x', 'g'), # gating x
    ('g02', 'h1', 'g'), # gating h1
    ('g03', 'o', 'g'), # gating o
    # ('g03', 't', 'g'),

    ('input', 'x', '1e'), # spikegen tp provide input to x
    # ('in_tgt', 't', '1e'),
    ('in_g', 'g00', '1ge'), # spikegen to initialize gating chain
]

gat_layers_inf = np.unique([p[0] for p in connected_pairs_inf if re.match("g[0-9][0-9]", p[0])])
num_gat_inf = max([int(l[1:]) for l in gat_layers_inf])+1
connected_pairs_inf += [("g" + str(i).rjust(2, '0'), "g" + str(i + 1).rjust(2, '0'), "1ge") for i in
                        range(num_gat_inf - 1)] + [("g" + str(num_gat_inf - 1).rjust(2, '0'), "g00", "1ge")]

layers_inf = {
    # layer_name : ('layer_type','neuron_type')
    # 't': ('out', 'n_sfc'),
    'input': ('in', 'n_inp'),
    # 'in_tgt': ('out', 'n_inp'),
    'in_g': ('gat', 'n_inp'),
    'g00': ('gat', 'n_gat'), 'g01': ('gat', 'n_gat'), 'g02': ('gat', 'n_gat'), 'g03': ('gat', 'n_gat'),
    'x': ('in', 'n_sfc'),
    'h1': ('hid', 'n_sfl'),
    'o': ('out', 'n_sfl'),
}

topology_inference = {'connected_pairs': connected_pairs_inf, 'layers': layers_inf}

# count layers:
count = {}
for l in layers:
    try:
        count[layers[l][0]] += 1
    except KeyError:
        count[layers[l][0]] = 1


if __name__ == '__main__':
    from SFC_backprop.network_parameters_2layer_MNIST import params

    num_populations = {}
    num_populations['hid'] = params['num_populations']['hid']
    num_populations['gat'] = 1
    num_populations['in'] = 100
    num_populations['out'] = 10

    conn_types = params['connection_types']

    count_conn = {}
    for p in connected_pairs:
        conn_type = conn_types[p[2]]['pop_conn_type']
        pre_type = layers[p[0]][0]
        post_type = layers[p[1]][0]

        try:
            count_conn[(conn_type, pre_type, post_type)] += 1
        except KeyError:
            count_conn[(conn_type, pre_type, post_type)] = 1

    all_neurons = 0
    print('Number of layers needed for network:')
    for l in count:
        print(l, count[l], 'total:', count[l] * num_populations[l])
        all_neurons += count[l] * num_populations[l]

    print('Number of connections needed for network:')
    all_syn = 0
    for c in count_conn:
        print(c, count_conn[c], end=', ')

        # pre_type,post_type = c[0].split(':')
        pre = c[1]
        post = c[2]
        if 'a' in c[0]:
            num_syn = num_populations[pre] * num_populations[post]
        else:
            num_syn = num_populations[pre]
            assert num_populations[pre] == num_populations[post]

        print('total:', num_syn * count_conn[c])
        all_syn += num_syn * count_conn[c]

    print('total number of neurons: ', all_neurons)
    print('total number of synapses: ', all_syn)
