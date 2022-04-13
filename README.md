© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

# sBP
Spiking backprop on Intel's research chip Loihi with gated synfire chain.

This is the code used in the following publication:
Renner, A., Sheldon, F., Zlotnik, A., Tao, L., & Sornborger, A. (2021). 
The Backpropagation Algorithm Implemented on Spiking Neuromorphic Hardware. 
arXiv preprint arXiv:2106.07030.
https://arxiv.org/abs/2106.07030

## Getting started
Please install nxsdk according to Intel's instructions.
For downloading and preparing the dataset, we also require the following packages:
mnist, scikit-image

The main file to run the code is SFC_backprop/SFC_backprop_main.py.

This code is written to allow rapid prototyping without writing code.



## Structure
In this section, the backend of the network is explained. You will not need this if
you just want to use the 3-layer network. However, you need it, if you want to change 
the network topology.

The main file to run the code is SFC_backprop/SFC_backprop_main.py.

The actual network is built in the BackpropNet class that in turn uses the more generic
ConnectedGroups.
ConnectedGroups constructs a network based on a topology and parameters.
A topology is a dictionary that contains `connected_pairs` and a layer mapping.
`connected_pairs` is a list of tuples of string identifiers for each connection in the whole network. E.g. 

    connected_pairs = [('x','h','p'),('h','o','p2')]

means that there is a connection of type p between layer x and h and a connection of type p2 between layer h and o.
The definition of what connection type `p` means, is given in the parameters.
In the layer mapping, each layer gets assigned a type.

    layers = {'x': ('in', 'n_sfc'),
              'h': ('hid', 'n_sfc'),
              'o': ('out', 'n_sfc')}

The first element of the tuple determines the layer type (e.g. how many neurons) and the second
element determines the neuron type (e.g. the specific neuron parameters).

Here, we are providing 2 pre-defined topologies from the paper:
A training network and an inference network.

Here, is the full topology of the 3-layer inference network:

    connected_pairs_inf = [
        ('x', 'h1', 'f'),  # W1 forward
        ('h1', 'o', 'f'),  # W2 forward
    
        ('g01', 'x', 'g'), # gating x 
        ('g02', 'h1', 'g'), # gating h1
        ('g03', 'o', 'g'), # gating o
    
        ('input', 'x', '1e'), # spikegen tp provide input to x
        ('in_g', 'g00', '1ge'), # spikegen to initialize gating chain

        ('g00', 'g01', '1ge'), #gating chain
        ('g01', 'g02', '1ge'),
        ('g02', 'g03', '1ge'),
        ('g03', 'g00', '1ge')
    ]
    layers_inf = {
        'input': ('in', 'n_inp'),
        'in_g': ('gat', 'n_inp'),
        'g00': ('gat', 'n_gat'),
        'g01': ('gat', 'n_gat'),
        'g02': ('gat', 'n_gat'),
        'g03': ('gat', 'n_gat'),
        'x': ('in', 'n_sfc'),
        'h1': ('hid', 'n_sfl'),
        'o': ('out', 'n_sfl')
    }

It only contains an input (x), hidden (h1) and output (o) layer, a gating neuron
for each layer (g00-03) and a spikegen layer to send in the inputs (input) and to kickstart
the gating chain (in_g).

The `ConnectedGroups` class takes the topology and loops over all layers to 
create `CompartmentGroup`s and over all connected pairs to create synapses.

In addition to the topology dictionary, a parameter dictionary is needed.
The parameter dictionary contains general network parameters and the 2 
sub-dictionaries `neuron_types` and `connection_types`.

`neuron_types` contains the neuron parameters, such as taus for each neuron type specified 
in the topology (e.g. 'n_sfc').

`connection_types` contains the connection parameters, such as the weight for each synapse 
type specified in the topology. Furthermore, it contains a connectivity type, such as 
`1:1` or `a:a` (for all-to-all connectivity) and a synapse class with which the synapse
is created. These synapse classes are usually wrappers around the `connect` funtions of
nxsdk.

The `ConnectedGroups` class is fully generic so that any kind of network can be created 
with it using the dictionary/string-based specification.
The `BackpropNet` class which inherits from `ConnectedGroups` is less generic and 
includes certain specificities of the backprop network.



