#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by : Alpha Renner (alpren@ini.uzh.ch)

num_neurons is the number of neurons per population
num_layers is the number of unique populations
num_populations is the number of parallel populations, e.g. if you have 4 inputs x1,x2,x3,x4, then this is 4.
"""

from SFC_backprop.loihi_groups import create_loihi_neuron, create_loihi_synapse, create_loihi_spikegen
from loihi_tools.weight_tools import calculate_mant_exp

params = {}

params['num_neurons'] = 1
params['num_trials'] = 60000 * 2  # we can do 2 epochs in one run, to save setup time

params['num_populations'] = {}
params['num_populations']['hid'] = 400  # 400 #400  # 400
params['num_populations']['gat'] = 1

params['weight_exponent'] = 0

params['T'] = 1

params['sfc_threshold'] = 1024  # 2 * 256 #2 * 256  # 4 * 256

bias = - 8192 * 64

params['x1TimeConstant'] = 1
params['y1TimeConstant'] = 1
params['x1Impulse'] = 0
params['y1Impulse'] = 0
params['r1TimeConstant'] = 0
params['r1Impulse'] = 1

sfc_neuron_params = {
    'tau_v': 1,
    'tau_i': 1,
    'threshold': params['sfc_threshold'],
    'refractory': 0,
    'i_const': bias,
    'enableNoise': 0,
    'enableLearning': 0,
    'num_neurons': params['num_neurons'],
    'neuron_creator': create_loihi_neuron,
    'start_core': 16,
    'end_core': 95
}
sfc_learn_neuron_params = {
    'tau_v': 1,
    'tau_i': 1,
    'threshold': params['sfc_threshold'],
    'refractory': 0,
    'i_const': bias,
    'enableNoise': 0,
    'enableLearning': 1,
    'num_neurons': params['num_neurons'],
    'neuron_creator': create_loihi_neuron,
    'start_core': 2,
    'end_core': 80
}
gating_neuron_params = {
    'tau_v': 1,
    'tau_i': 1,
    'threshold': params['sfc_threshold'],
    'refractory': 0,
    'i_const': 0,
    'num_neurons': params['num_neurons'],
    'neuron_creator': create_loihi_neuron,
    'start_core': 40,
    'end_core': 95
}
reward_neuron_params = {
    'tau_v': 1,
    'tau_i': 1,
    'threshold': params['sfc_threshold'],
    'refractory': 0,
    'i_const': 0,
    'num_neurons': params['num_neurons'],
    'neuron_creator': create_loihi_neuron,
    'start_core': 0,
    'end_core': 1
}
input_neuron_params = {
    'neuron_creator': create_loihi_spikegen
}

params['neuron_types'] = {
    'n_sfc': sfc_neuron_params,
    'n_sfl': sfc_learn_neuron_params,
    # 'n_ref': sfc_ref_neuron_params,
    'n_gat': gating_neuron_params,
    'n_rew': reward_neuron_params,
    'n_inp': input_neuron_params,
}

tau = sfc_neuron_params['tau_i']
# p_synapse = params['p_synapse']
num_neurons = params['num_neurons']

binary_threshold = params['sfc_threshold'] // 2  # This is the value at which the neuron activity is rounded up
assert binary_threshold == params['sfc_threshold'] / 2
params['weight_p'] = params['sfc_threshold']  # This is the max weight

w1e, exp1e = calculate_mant_exp(binary_threshold, verbose=0)
params['weight_1e'] = binary_threshold + 2 ** exp1e  # This is done to avoid rounding errors
w1ge, exp1ge = calculate_mant_exp(params['sfc_threshold'] - bias // 64, verbose=0)
params['weight_1ge'] = -bias // 64 + params['sfc_threshold'] + 2 ** exp1ge

params['weight_i'] = -params['sfc_threshold']
params['weight_e'] = params['sfc_threshold']
params['weight_g'] = -bias // 64 + binary_threshold
# - 2  # Noise goes all the way to the threshold, so here we set the gate weight
params['weight_gp05'] = -bias // 64
params['weight_gm05'] = binary_threshold  # +4 # - 2

params['weight_gi'] = -254 * 2 ** 6  # TODO

syn_delay = 0  # 1 means delay of 2 ...!?

# identifier: (population connectivity, layer connectivity, connection probability)
params['connection_types'] = {
    'p': {'pop_conn_type': 'a:a',
          'syn': create_loihi_synapse,
          'params': {'weight': 0,
                     'weight_exponent': params['weight_exponent'],
                     # 'weight_factor': params['weight_p'],
                     'delay': syn_delay,
                     'x1TimeConstant': params['x1TimeConstant'],
                     'y1TimeConstant': params['y1TimeConstant'],
                     'r1TimeConstant': params['r1TimeConstant'],
                     'x1Impulse': params['x1Impulse'],
                     'y1Impulse': params['y1Impulse'],
                     'r1Impulse': params['r1Impulse'],
                     'lr_w': 'y0*x0*r1*4 - y0*x0*2',  # TODO
                     # 'lr_t': 'x0*r1*0'
                     }
          },
    'f': {'pop_conn_type': 'a:a',
          'syn': create_loihi_synapse,
          'params': {'weight': 0,
                     'weight_exponent': params['weight_exponent'],
                     'delay': syn_delay,
                     # 'lr_w': '0*x0',
                     }
          },
    '1e': {'pop_conn_type': '1:1',
           'syn': create_loihi_synapse,
           'params': {'weight': params['weight_1e'],
                      'delay': syn_delay,
                      'weight_factor': 1,
                      }
           },
    '1e_d1': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 1,
                         'weight_factor': 1,
                         }
              },
    '1e_d2': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 2,
                         'weight_factor': 1,
                         }
              },
    '1e_d3': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 3,
                         'weight_factor': 1,
                         }
              },
    '1e_d4': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 4,
                         'weight_factor': 1,
                         }
              },
    '1e_d5': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 5,
                         'weight_factor': 1,
                         }
              },
    '1e_d6': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_1e'],
                         'delay': 6,
                         'weight_factor': 1,
                         }
              },
    '1ge': {'pop_conn_type': '1:1',
            'syn': create_loihi_synapse,
            'params': {'weight': params['weight_1ge'],
                       'delay': syn_delay,
                       'weight_factor': 1,
                       }
            },
    '1ge_d2': {'pop_conn_type': '1:1',
               'syn': create_loihi_synapse,
               'params': {'weight': params['weight_1ge'],
                          'delay': 2,
                          'weight_factor': 1,
                          }
               },
    '1i': {'pop_conn_type': '1:1',
           'syn': create_loihi_synapse,
           'params': {'weight': -1 * params['weight_1e'],
                      'delay': syn_delay,
                      'weight_factor': 1,
                      }
           },

    '1i_d3': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': -1 * params['weight_1e'],
                         'delay': 3,
                         'weight_factor': 1,
                         }
              },
    'g': {'pop_conn_type': '1:a',
          'syn': create_loihi_synapse,
          'params': {'weight': params['weight_g'],
                     'delay': syn_delay,
                     'weight_factor': 1,
                     }
          },
    'g1': {'pop_conn_type': '1:1',
           'syn': create_loihi_synapse,
           'params': {'weight': params['weight_g'],
                      'delay': syn_delay,
                      'weight_factor': 1,
                      }
           },
    'g1_d4': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_g'],
                         'delay': 4,
                         'weight_factor': 1,
                         }
              },
    'g1s': {'pop_conn_type': '1:1',
            'syn': create_loihi_synapse,
            'params': {'weight': params['weight_g'],
                       'delay': syn_delay,
                       'weight_factor': 1,
                       }
            },
    'gi': {'pop_conn_type': '1:a',
           'syn': create_loihi_synapse,
           'params': {'weight': params['weight_gi'],
                      'delay': syn_delay,  # 1 * sim_dt,  # 1 means no delay
                      'weight_factor': 1,
                      }
           },
    'gi2': {'pop_conn_type': '1:a',
            'syn': create_loihi_synapse,
            'params': {'weight': params['weight_gi'],
                       'delay': syn_delay,  # 1 * sim_dt,  # 1 means no delay
                       'weight_factor': 1,
                       }
            },
    'gp05': {'pop_conn_type': '1:a',
             'syn': create_loihi_synapse,
             'params': {'weight': params['weight_gp05'],
                        'delay': syn_delay,  # 1 * sim_dt,  # 1 means no delay
                        'weight_factor': 1,
                        }
             },
    'gm05': {'pop_conn_type': '1:a',
             'syn': create_loihi_synapse,
             'params': {'weight': params['weight_gm05'],
                        'delay': syn_delay,  # 1 * sim_dt,  # 1 means no delay
                        'weight_factor': 1,
                        }
             },
    'g1m05': {'pop_conn_type': '1:1',
              'syn': create_loihi_synapse,
              'params': {'weight': params['weight_gm05'],  # -8,
                         'delay': syn_delay,  # 1 * sim_dt,  # 1 means no delay
                         'weight_factor': 1,
                         }
              },
}
