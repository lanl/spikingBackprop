import numpy as np
import warnings

try:
    import nxsdk.api.n2a as nx
except:
    warnings.warn('nxsdk not found')


def calculate_effective_weight(numWeightBits=6, IS_MIXED=0, weight=255, weightExponent=0):
    '''
    calculates and prints the actual weight of a synapse after discretization
    Please compare with the following documentation file: /docs/connection.html
    :param numWeightBits: number of  weight bits
    :param IS_MIXED: is the sign mode mixed (1)? Set to 0 if sign mode is exc or inh.
    :param weight: the weight
    :param weightExponent: weight exponent
    :return: the effective weight
    '''
    numLsbBits = 8 - (numWeightBits - IS_MIXED)
    actWeight = (weight >> numLsbBits) << numLsbBits
    print('original weight:', weight)
    print('actual weight:', actWeight)
    print('num lsb:', numLsbBits)
    print('weight (with exponent):', actWeight * 2 ** weightExponent)
    print('weight effect on current (with exponent):', actWeight * 2 ** (6 + weightExponent))
    return actWeight


def calculate_mant_exp(value, precision=8, name="", exponent_bounds=(-8, 7), weight_precision=7, verbose=True):
    """
    This function calculates the exponent and mantissa from a desired value. Can e.g. be used for weights.
    If used for weights: Please use calculate_effective_weight to calculate
     the effective weight also taking into account the precision.

    Important: This is based on a very simple idea to maximize the precision.
     However, it does not replace manual checking of your weights
     (E.g. instead of letting them go from 0-300, you should restrict them to 0-255,
     as you will loose precision otherwise)

    Also, be careful when using this with plastic weights.
    Otherwise your range might be limited to the initial weight range.

    the given exponent bounds [-8, 7] are the default for weights on loihi

    Note that the negative exponent bound will be overwritten if it is too low. It should not be lower
    than 8 - precision, as the final result (weight * 2**exponent) still has to be integer for integration
    on the membrane potential. I.e. lower weight exponents than 8-precision are useless (and have the potential
    to be misinterpreted).

    :param value: the value for which you want to calculate mantissa and exponent
    :param precision: the allowed precision in bits for the mantissa (this actually needs to be 8 for synapses
        independent of the weight precision, as the value always goes to 255)
    :param name: just for printing
    :param exponent_bounds: bounds of the exponent. E.g. on Loihi, the weight exponent can range from -8 to +7
    :return: mantissa, exponent
    """
    value = np.asarray(value)
    if verbose:
        print('desired value:', value)
    val_des = value
    exponent = 0
    while np.max(np.abs(value)) >= (2 ** precision) and not exponent == exponent_bounds[1]:
        value = value / 2
        exponent += 1

    exponent_bounds = np.asarray(exponent_bounds)
    exponent_bounds[0] = np.max([exponent_bounds[0], weight_precision - 8])

    while np.abs(np.max(value)) < (2 ** precision / 2) and np.abs(np.max(value)) != 0 and not exponent == \
                                                                                              exponent_bounds[0]:
        value = value * 2
        exponent += -1

    value = np.asarray(np.round(value), dtype=int)
    if verbose:
        print('actual value of', name, ':', value * 2 ** exponent, 'mantissa:', value, 'exponent:', exponent)
    if (val_des != (value * 2 ** exponent)).any():
        if verbose:
            print('rounding error for ' + name + '!')
        warnings.warn('rounding error for ' + name + '!')
    return value, exponent


def create_conn_prototype(weight_matrix, weight_exponent=None, verbose=True, **kwargs):
    scalar = False
    if not hasattr(weight_matrix, "__iter__"):
        weight_matrix = np.asarray([weight_matrix])
        scalar = True
    elif type(weight_matrix) == list:
        weight_matrix = np.asarray(weight_matrix)

    try:
        weight_matrix = weight_matrix.tocsr()
        max_abs = abs(weight_matrix).max()
    except:
        # sign_mat = np.sign(weight_matrix)
        max_abs = np.max(np.abs(weight_matrix))

    max_val = np.max(weight_matrix)
    min_val = np.min(weight_matrix)

    if max_val > 0 and min_val < 0:
        signMode = nx.SYNAPSE_SIGN_MODE.MIXED
        weight_precision = 7
    elif min_val > 0:
        signMode = nx.SYNAPSE_SIGN_MODE.EXCITATORY
        weight_precision = 8
    elif max_val < 0:
        signMode = nx.SYNAPSE_SIGN_MODE.INHIBITORY
        weight_precision = 8
    else:  # This important, as we might initialize plastic weights with 0 and they are usually mixed
        signMode = nx.SYNAPSE_SIGN_MODE.MIXED
        weight_precision = 7

    if weight_exponent is None:
        _, weight_exponent = calculate_mant_exp(max_abs, precision=8, name="weight", weight_precision=weight_precision,
                                                verbose=verbose)
    weight_matrix = weight_matrix * 2 ** (-weight_exponent)
    weight_matrix = np.asarray(np.round(weight_matrix), dtype=int)
    if verbose:
        print('max weight: ', np.max(weight_matrix))
        print('min weight: ', np.min(weight_matrix))

    try:
        numDelayBits = kwargs['numDelayBits']
    except KeyError:
        kwargs['numDelayBits'] = 2

    try:
        numDelayBits = kwargs['numWeightBits']
    except KeyError:
        kwargs['numWeightBits'] = 8

    try:
        numDelayBits = kwargs['numTagBits']
    except KeyError:
        kwargs['numTagBits'] = 1

    prototype = nx.ConnectionPrototype(signMode=signMode,
                                       weightExponent=weight_exponent, **kwargs)

    if scalar:
        weight_matrix = weight_matrix[0]

    return prototype, weight_matrix


if __name__ == '__main__':
    calculate_effective_weight(numWeightBits=4, IS_MIXED=0, weight=250, weightExponent=3)
