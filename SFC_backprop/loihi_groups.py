#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np
from nxsdk.net.process.basicspikegen import BasicSpikeGen

from loihi_tools.compartment_tools import create_distributed_group_over_cores, create_compartment_prototype
from loihi_tools.spikegenerators import create_spikegen
from loihi_tools.weight_tools import calculate_mant_exp, create_conn_prototype


def create_loihi_neuron(net, N, parameters=None, name='neuron', verbose=True):
    try:
        parameters['i_const']
    except KeyError:
        parameters['i_const'] = 0

    try:
        parameters['refractory']
    except KeyError:
        parameters['refractory'] = 0

    try:
        parameters['enableLearning']
    except KeyError:
        parameters['enableLearning'] = 0

    try:
        parameters['noiseMantAtCompartment'] = parameters['noiseMantAtCompartment']
        parameters['noiseExpAtCompartment'] = parameters['noiseExpAtCompartment']
        parameters['enableNoise'] = parameters['enableNoise']
    except KeyError:
        parameters['noiseMantAtCompartment'] = 0
        parameters['noiseExpAtCompartment'] = 0
        parameters['enableNoise'] = 0

    bias_mant, bias_exp = calculate_mant_exp(parameters['i_const'], precision=8,
                                             name=name + "_i_const", verbose=verbose)

    if parameters['enableLearning']:
        if verbose:
            print(name, "has enableLearning == 1")
        comp_prototype = create_compartment_prototype(compartmentVoltageTimeConstant=parameters['tau_v'],
                                                      compartmentCurrentTimeConstant=parameters['tau_i'],
                                                      vTh=parameters['threshold'] * 64,
                                                      refractoryDelay=max(parameters['refractory'], 1),
                                                      biasMant=bias_mant,
                                                      biasExp=bias_exp,
                                                      enableSpikeBackprop=1,
                                                      enableSpikeBackpropFromSelf=1,
                                                      noiseMantAtCompartment=parameters['noiseMantAtCompartment'],
                                                      noiseExpAtCompartment=parameters['noiseExpAtCompartment'],
                                                      enableNoise=parameters['enableNoise'],
                                                      randomizeVoltage=parameters['enableNoise'],
                                                      numDendriticAccumulators=8,
                                                      verbose=verbose
                                                      # tEpoch=1
                                                      )
        # TODO: When we set randomizeVoltage=1, it only works for V noise, not I noise
    else:
        comp_prototype = create_compartment_prototype(compartmentVoltageTimeConstant=parameters['tau_v'],
                                                      compartmentCurrentTimeConstant=parameters['tau_i'],
                                                      vTh=parameters['threshold'] * 64,
                                                      refractoryDelay=max(parameters['refractory'], 1),
                                                      biasMant=bias_mant,
                                                      biasExp=bias_exp,
                                                      noiseMantAtCompartment=parameters['noiseMantAtCompartment'],
                                                      noiseExpAtCompartment=parameters['noiseExpAtCompartment'],
                                                      enableNoise=parameters['enableNoise'],
                                                      randomizeVoltage=parameters['enableNoise'],
                                                      numDendriticAccumulators=8,
                                                      verbose=verbose
                                                      )

    loihi_group = create_distributed_group_over_cores(net=net, name=name, size=N,
                                                      prototype=comp_prototype,
                                                      start_core=parameters['start_core'],
                                                      end_core=parameters['end_core'],
                                                      verbose=verbose)

    return loihi_group


def create_loihi_synapse(net, source, target, conn_parameters, mask, name, verbose=True):
    """
    Args:
        source (NeuronGroup, Neurons obj.): Pre-synaptic neuron population.
        target (NeuronGroup, Neurons obj.): Post-synaptic neuron population.
        name (str, optional): Name of synapse group.

    """

    pre_compartments = source
    post_compartments = target

    try:
        if verbose:
            print('learning rule is:', conn_parameters['lr_w'])
            print("lr params are:")
            print(conn_parameters['x1Impulse'], conn_parameters['x1TimeConstant'])
        try:
            lr_t = conn_parameters['lr_t']
        except KeyError:
            lr_t = None
        lr = net.createLearningRule(dw=conn_parameters['lr_w'],
                                    dt=lr_t,
                                    x1Impulse=conn_parameters['x1Impulse'],
                                    x1TimeConstant=conn_parameters['x1TimeConstant'],
                                    y1Impulse=conn_parameters['y1Impulse'],
                                    y1TimeConstant=conn_parameters['y1TimeConstant'],
                                    r1Impulse=conn_parameters['r1Impulse'],
                                    r1TimeConstant=conn_parameters['r1TimeConstant'],
                                    tEpoch=1)
        enableLearning = 1
        learningRule = lr
        if verbose:
            print(lr.reinforcementChannel)
    except KeyError:
        enableLearning = 0
        learningRule = None
        weight_exponent = None

    if conn_parameters['delay'] > 0:
        disableDelay = 0
        delay = np.round(conn_parameters['delay'])
        if verbose:
            print("delay enabled for", name, ':', delay)
        # bin_str = bin(delay+2)[3:]
        # numDelayBits = len(bin_str)
        numDelayBits = 3
    else:
        disableDelay = 1
        delay = 0
        numDelayBits = 0
        if verbose:
            print("delay disabled for", name)

    if isinstance(pre_compartments, BasicSpikeGen):
        num_pre = pre_compartments.numPorts
    else:
        num_pre = pre_compartments.numNodes

    # this is the same as simple integer rounding (just wanted to be sure to get integers)
    weight_r = np.asarray(np.sign(conn_parameters['weight']), dtype=int) * \
               np.asarray(np.abs(conn_parameters['weight']) + 0.5, dtype=int)
    if not np.sum(weight_r - conn_parameters['weight']) == 0:
        warnings.warn("rounding error for weight init of " + name)

    # weight_r = np.asarray(np.sign(conn_parameters['weight']), dtype=int) * \
    #            np.asarray(np.abs(conn_parameters['weight'] * conn_parameters['w_factor']) + 0.5, dtype=int)
    # if not np.sum(weight_r - conn_parameters['weight'] * conn_parameters['w_factor']) == 0:
    #     warnings.warn("rounding error for weight init of " + name)

    try:
        weight_exponent = conn_parameters['weight_exponent']
        if verbose:
            print("weight_exponent set for", name, weight_exponent)
    except KeyError:
        weight_exponent = None

    conn_proto, weight_matrix = create_conn_prototype(
        weight_matrix=np.asarray(weight_r),
        weight_exponent=weight_exponent,
        enableLearning=enableLearning, learningRule=learningRule,
        disableDelay=disableDelay, delay=delay, numDelayBits=numDelayBits,
        verbose=verbose
    )

    if verbose:
        print(weight_matrix)
    # print(mask)
    conn_group = pre_compartments.connect(post_compartments, prototype=conn_proto,
                                          weight=weight_matrix,
                                          connectionMask=mask)

    if verbose:
        try:
            print("created connection from", pre_compartments.name, "to", post_compartments.name)
        except AttributeError:
            print("created connection from somewhere (probably spikegen) to", post_compartments.name)
            # print("created connection from", pre_compartments, "to", post_compartments)
        # print("weight", weight_matrix)
        # print("mask:", mask)

    return conn_group


def create_loihi_spikegen(net, N, parameters, name, verbose=True):
    if verbose:
        print('create spikegen:', name)
    spikegen = create_spikegen(net, indices=np.asarray([]), spiketimes=np.asarray([]), numPorts=N, verbose=verbose)
    return spikegen


def calc_spiketimes_from_input_arr(input_arr, interval, max_rate, num_neurons, T):
    indices = []
    times = []
    for input_layer, input_data in enumerate(input_arr.T):
        # print(input_layer)
        for trial in np.where(input_data)[0]:
            input_rate = int(max_rate * input_data[trial])
            input_time = trial * interval + 1
            if num_neurons == 1:
                indices += [input_layer * num_neurons]
            else:
                indices += list(input_layer * num_neurons + np.random.choice(range(num_neurons),
                                                                             size=input_rate, replace=False))
            if T == 1:
                times += [input_time] * input_rate
            else:
                times += list(input_time + np.random.choice(range(T), size=input_rate, replace=True))

    return indices, times
