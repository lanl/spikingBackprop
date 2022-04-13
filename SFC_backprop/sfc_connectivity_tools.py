#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def get_neuron_conn_indices_by_type(connected_pairs, connection_types,
                                    layer_map_pre, layer_map_post,
                                    num_neurons_pre, num_neurons_post):
    connection_types_present = np.unique([pair[2] for pair in connected_pairs])
    neuron_conn_indices = {}
    for conn_type in connection_types_present:
        conn_params = connection_types[conn_type]
        pop_conn_type = conn_params['pop_conn_type']
        lay_conn_type = conn_params['lay_conn_type']
        p = conn_params['p']

        # which layers have to be connected to which?
        # layer_conn_indices = _get_layer_conn_indices_from_pairs(layer_map_pre, layer_map_post, connected_pairs,
        #                                                         conn_type)

        # which layers have to be connected?
        connected_pairs_type = [[pre_lay, post_lay] for pre_lay, post_lay, ctype in connected_pairs if
                                ctype == conn_type]
        connected_layers = []
        for pair in connected_pairs_type:
            check = 1  # if we want to make sure that all connections make sense
            if check:
                num_populations_pre = np.sum([1 for k in layer_map_pre if k[0] == pair[0]])
                num_populations_post = np.sum([1 for k in layer_map_post if k[0] == pair[1]])
                # print(conn_params)
                # print(num_populations_post, num_populations_pre)
                if pop_conn_type == '1:a':
                    assert num_populations_pre == 1, print(pair)
                if pop_conn_type == 'a:1':
                    assert num_populations_post == 1, print(pair)
                if pop_conn_type == '1:1':
                    assert num_populations_post == num_populations_pre, print(pair)
            if pop_conn_type == '1:1':
                connected_layers += [[k_pre, k_post] for k_post in layer_map_post if k_post[0] == pair[1]
                                     for k_pre in layer_map_pre if k_pre[0] == pair[0] and k_pre[1] == k_post[1]]
            else:
                connected_layers += [[k_pre, k_post] for k_post in layer_map_post if k_post[0] == pair[1]
                                     for k_pre in layer_map_pre if k_pre[0] == pair[0]]

        connected_layer_indices = [[layer_map_pre[pre], layer_map_post[post]] for pre, post in connected_layers]

        # which neurons have to be connected to which?
        neuron_conn_indices[conn_type] = _get_neuron_conn_indices(connected_layer_indices,
                                                                  num_neurons_pre, num_neurons_post,
                                                                  lay_conn_type=lay_conn_type, p=p,
                                                                  visualize=False)

    return neuron_conn_indices


# def _get_layer_conn_indices_from_pairs(layer_map_pre, layer_map_post, connected_pairs, conn_name):
#     lay_conn_indices = [[layer_map_pre[pre_lay], layer_map_post[post_lay]] for pre_lay, post_lay, conn_type in
#                         connected_pairs if conn_name == conn_type]
#     return np.asarray(lay_conn_indices).T


def _get_neuron_conn_indices(connected_layer_indices,
                             num_neurons_pre, num_neurons_post,
                             lay_conn_type='a:a', p=1,
                             visualize=False):
    """
    creates indices for an all to all connectivity between several populations in a single neuron group
     or in different neuron groups.
     For now, we assume num_neurons_i = num_neurons_j, need to do some testing if not!
    """

    if lay_conn_type == '1:a':
        assert num_neurons_pre == 1
        lay_conn_type = 'a:a'
    if lay_conn_type == 'a:1':
        assert num_neurons_post == 1
        lay_conn_type = 'a:a'

    if len(connected_layer_indices) > 0:
        jjs = []
        iis = []
        for pre, post in connected_layer_indices:
            if lay_conn_type == '1:1':
                ii, jj = _one_to_one(pre, post, num_neurons_pre, p=1)
            elif lay_conn_type == 'a:a':
                ii, jj = _all_to_all(pre, post, num_neurons_pre, num_neurons_post, p=p)
            else:
                raise NotImplementedError('lay_conn_type not supported')
            iis += list(ii)
            jjs += list(jj)

        iis = np.asarray(iis)
        jjs = np.asarray(jjs)

        if visualize:
            skip = 1
            sourceNeuron = max(max(iis), max(jjs))
            targetNeuron = max(max(iis), max(jjs))
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.plot(np.zeros(sourceNeuron), range(sourceNeuron), 'ok', ms=10)
            plt.plot(np.ones(targetNeuron), range(targetNeuron), 'ok', ms=10)
            for i, j in zip(iis[::skip], jjs[::skip]):
                plt.plot([0, 1], [i, j], '-k')
            plt.xticks([0, 1], ['pre', 'post'])
            plt.ylabel('Neuron index')
            plt.xlim(-0.1, 1.1)
            plt.ylim(-1, max(sourceNeuron, targetNeuron))
            plt.subplot(122)
            plt.plot(iis, jjs, 'ok')
            plt.xlim(-1, sourceNeuron)
            plt.ylim(-1, targetNeuron)
            plt.xlabel('Source neuron index')
            plt.ylabel('Target neuron index')
    else:
        iis = np.asarray([], dtype=int)
        jjs = np.asarray([], dtype=int)
        # print(connected_layer_indices)
        import warnings
        warnings.warn('conn indices are empty')

    return iis, jjs


def _all_to_all(from_layer, to_layer, num_neurons_i, num_neurons_j, p=1):
    # pre connection indices
    iis = np.zeros((num_neurons_i * num_neurons_j), dtype=int)
    # range from 0 to num_neurons_i for every neuron, e.g. 0 0 0 1 1 1 2 2 2 3 3 3
    iis[:] = np.repeat(np.arange(0, num_neurons_i, dtype=int), num_neurons_j)
    iis[:] = iis[:] + from_layer * num_neurons_i

    jjs = np.zeros((num_neurons_i * num_neurons_j), dtype=int)
    # range from 0 to num_neurons_j tiled for every neuron, e.g. 0 1 2 3 0 1 2 3 0 1 2 3
    jjs[:] = np.tile(np.arange(0, num_neurons_j, dtype=int), num_neurons_i)
    # jjs[:] = jjs[:] + (pop_indices * num_neurons_i * num_neurons_j)[np.newaxis, :]
    jjs[:] = jjs[:] + (to_layer * num_neurons_j)

    num_conns = num_neurons_i * num_neurons_j
    num_selected = int(p * num_conns)
    chosen_idx = sorted(np.random.choice(range(0, num_conns), num_selected, replace=False))
    iis = iis[chosen_idx]
    jjs = jjs[chosen_idx]

    return iis, jjs


def _one_to_one(from_layer, to_layer, num_neurons, p=1):
    # pre connection indices
    # iis = np.zeros((num_neurons), dtype=int)
    # range from 0 to num_neurons_i for every neuron, e.g. 0 0 0 1 1 1 2 2 2 3 3 3
    iis = np.arange(0, num_neurons, dtype=int)
    iis[:] = iis[:] + from_layer * num_neurons

    # jjs = np.zeros((num_neurons), dtype=int)
    # range from 0 to num_neurons_j tiled for every neuron, e.g. 0 1 2 3 0 1 2 3 0 1 2 3
    jjs = np.arange(0, num_neurons, dtype=int)
    # jjs[:] = jjs[:] + (pop_indices * num_neurons_i * num_neurons_j)[np.newaxis, :]
    jjs[:] = jjs[:] + to_layer * num_neurons

    num_conns = num_neurons
    num_selected = int(p * num_conns)
    chosen_idx = sorted(np.random.choice(range(0, num_conns), num_selected, replace=False))
    iis = iis[chosen_idx]
    jjs = jjs[chosen_idx]

    return iis, jjs
