import numpy as np


def create_spikegen(net, indices, spiketimes, numPorts=None, verbose=True):
    """
    creates a spikegenerator from a list of indices with corresponding spiketimes
    :param net: the nx net in which the spikegen is created
    :param indices: the neuron indices that correspond to the spiketimes
    :param spiketimes: times at which the spikegen spikes
    :param numPorts: the numebr of spikegen neurons, if None, it is the max of indices
    :return: spikegenerator (needed to connect to other groups)
    """
    if numPorts is None:
        numPorts = np.max(indices)
    if len(spiketimes) > 0:
        if verbose:
            print('length of stimulation is', np.max(spiketimes), 'timesteps')
    else:
        if verbose:
            print('empty spikegen')
    spikegen = net.createSpikeGenProcess(numPorts=numPorts)
    for sg_neuron in np.unique(np.asarray(indices)):
        spikegen.addSpikes(spikeInputPortNodeIds=sg_neuron,
                           spikeTimes=list(spiketimes[np.where(indices == sg_neuron)]))
    return spikegen


def add_spikes_to_spikegen(spikegen, indices, spiketimes, verbose=True):
    """
    adds spikes from a list of indices with corresponding spiketimes
    :param spikegen: the spikegen to add spikes to
    :param indices: the neuron indices that correspond to the spiketimes
    :param spiketimes: times at which the spikegen spikes
    """
    numPorts = spikegen.numPorts
    if len(spiketimes) > 0:
        if verbose:
            print('length of stimulation is', np.max(spiketimes), 'timesteps')
    else:
        if verbose:
            print('empty spikegen')
        return

    spiketimes = np.asarray(spiketimes)
    for sg_neuron in np.unique(np.asarray(indices)):
        spikegen.addSpikes(spikeInputPortNodeIds=sg_neuron,
                           spikeTimes=list(spiketimes[(indices == sg_neuron)]))
