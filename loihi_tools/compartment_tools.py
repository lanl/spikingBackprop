import numpy as np
import nxsdk.api.n2a as nx


def create_compartment_prototype(compartmentVoltageTimeConstant, compartmentCurrentTimeConstant, vTh,
                                 refractoryDelay=None, verbose=True,
                                 **CompartmentPrototype_kwargs):
    """
    Wrapper for nx.CompartmentPrototype that allows direct creation of the prototype with taus instead of decay
    Note that threshold is divided by 64 in order to compensate for the fact that loihi multiplies by 64

    :param compartmentVoltageTimeConstant: voltage tau
    :param compartmentCurrentTimeConstant: current tau
    :param vTh: threshold
    :param refractoryDelay: refractory period
    :param CompartmentPrototype_kwargs:
    :return: CompartmentPrototype
    """
    compartmentVoltageDecay = int(4096 / compartmentVoltageTimeConstant)
    compartmentCurrentDecay = int(4096 / compartmentCurrentTimeConstant)
    if verbose:
        print('vTh: ', vTh)
    # print(refractoryDelay)

    prototype = nx.CompartmentPrototype(vThMant=vTh // 64,
                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                        compartmentVoltageDecay=compartmentVoltageDecay,
                                        compartmentCurrentDecay=compartmentCurrentDecay,
                                        refractoryDelay=refractoryDelay // 1,
                                        # enableHomeostasis=1, homeostasisGain=0,
                                        # activityImpulse=1, activityTimeConstant=0,
                                        # tEpoch=1,
                                        **CompartmentPrototype_kwargs
                                        )
    if verbose:
        print("create compartment group with the following parameters:")
        print("I and V decay:",
              prototype.compartmentCurrentDecay,
              prototype.compartmentVoltageDecay,
              "I and V tau:",
              prototype.compartmentCurrentTimeConstant,
              prototype.compartmentVoltageTimeConstant,
              "bias:",
              prototype.bias,
              prototype.biasMant,
              prototype.biasExp,
              "compartmentThreshold:",
              prototype.compartmentThreshold,
              "functionalState",
              prototype.functionalState,
              "refractoryDelay",
              prototype.refractoryDelay,
              )

    return prototype


def create_distributed_group_over_cores(net, start_core, end_core, name, prototype, size, verbose=True):
    """
    distribute Neurons over cores
    :param net:
    :param start_core:
    :param end_core:
    :param name:
    :param prototype:
    :param size:
    :return:
    """
    if verbose:
        print("create compartment group with the following parameters:")
        print("I and V decay:",
              prototype.compartmentCurrentDecay,
              prototype.compartmentVoltageDecay,
              "I and V tau:",
              prototype.compartmentCurrentTimeConstant,
              prototype.compartmentVoltageTimeConstant,
              "bias:",
              prototype.bias,
              prototype.biasMant,
              prototype.biasExp,
              "compartmentThreshold:",
              prototype.compartmentThreshold,
              "functionalState",
              prototype.functionalState,
              "refractoryDelay",
              prototype.refractoryDelay,
              )

    num_neurons = size
    group = net.createCompartmentGroup(name, size=0, prototype=prototype)
    num_cores = end_core - start_core
    neurons_per_core = np.ceil(num_neurons / num_cores)
    if verbose:
        print(name, neurons_per_core, 'neurons per core')
        print(num_neurons, 'neurons from', start_core, 'to', end_core)
    for i in range(num_neurons):
        core_id = start_core + (i // neurons_per_core)
        # core_id = start_core + (i % neurons_per_core)
        prototype.logicalCoreId = core_id
        comp = net.createCompartment(prototype=prototype)
        group.addCompartments(comp)

    return group
