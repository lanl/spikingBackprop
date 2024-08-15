#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by : Alpha Renner (alpren@ini.uzh.ch)

"""
import os
import re
import time
import warnings

import numpy as np
import nxsdk.api.n2a as nx

from SFC_backprop.input_data import generate_input_data
from SFC_backprop.loihi_groups import calc_spiketimes_from_input_arr
from SFC_backprop.network_topology_2layer import topology_inference, topology_learn
from SFC_backprop.simulation import inference_from_weights
from SFC_backprop.synfire_chain import ConnectedGroups
from SFC_backprop.weight_init import weight_init
from loihi_tools.spikegenerators import add_spikes_to_spikegen


class BackpropNet(ConnectedGroups):
    """
    this is a wrapper for ConnectedGroups with specific parameters etc. for the backprop network
    It handles inputs, probes, etc.
    """

    def __init__(self, params, debug=0):
        self.monitor_layers = []
        if debug > 0:
            self.verbose = True
        else:
            self.verbose = False

        self.params = params
        self.do_train = params['do_train']
        self.do_probe_energy = params['do_probe_energy']
        self.on_kapohobay = params['on_kapohobay']
        self.weight_mode = params['weight_mode']
        self.weight_file = params['weight_file']
        self.dataset = params['dataset']
        try:
            self.input_binary_threshold = params['input_binary_threshold']
        except KeyError:
            self.input_binary_threshold = 0.5

        if self.do_train:
            topology = topology_learn
        else:
            topology = topology_inference

        num_gate = len(np.unique([k for k in topology['layers'] if re.match("g[0-9][0-9]", k)]))
        params['num_gate'] = num_gate
        num_trials = params['num_trials']
        num_steps = num_gate * num_trials + num_gate + 1
        params['num_steps'] = num_steps

        if self.do_probe_energy:
            if self.on_kapohobay:
                os.environ["KAPOHOBAY"] = "1"
                import nxsdk
                import subprocess

                server_path = os.path.join(nxsdk.__path__[0], 'telemetry', 'kapoho_server.py')
                command = ('python ' + server_path).split()
                self.telemetry_server_process = subprocess.Popen(command, stdout=subprocess.PIPE)
            else:
                os.environ['SLURM'] = "1"
                os.environ['PARTITION'] = "nahuku32"
                os.environ['BOARD'] = "ncl-ext-ghrd-01"
        else:
            if self.on_kapohobay:
                os.environ["KAPOHOBAY"] = "1"

        print("generating input data...")

        if num_trials <= 60000:
            input_data, target_data = generate_input_data(num_trials, input_data=self.dataset,
                                                          add_bias=False, threshold=self.input_binary_threshold)
        else:
            input_data, target_data = generate_input_data(60000, input_data=self.dataset,
                                                          add_bias=False, threshold=self.input_binary_threshold)

        self.input_data = input_data
        self.target_data = target_data

        params['num_populations']['in'] = input_data.shape[1]
        params['num_populations']['out'] = target_data.shape[1]

        net = nx.NxNet()
        ConnectedGroups.__init__(self, net, topology=topology, params=params, verbose=self.verbose)

        print("initializing weights...")
        init_weight_matrix0, init_weight_matrix1 = weight_init(params, mode=self.weight_mode,
                                                               file=self.weight_file)

        if self.do_train:
            plastic_connection_map = {
                'w1': 's_x_h1_p',
                'w1_copy1': 's_x_h1_copy_p',
                'w1_copy2': 's_x_h1_copy2_p',
                'w2': 's_h1_o_p',
                'w2_copy1': 's_h1_o_copy_p',
                'w2_copy2': 's_h1_o_copy2_p',
                'w2Tp': 's_o_h1T_p',
                'w2Tm': 's_oT-_h1T_p'
            }
        else:
            plastic_connection_map = {
                'w1': 's_x_h1_f',
                'w2': 's_h1_o_f',
            }
        self.plastic_connection_map = plastic_connection_map

        # Init weight matrices
        for weight_name in plastic_connection_map:
            if 'w1' in weight_name:
                init_mat = init_weight_matrix0
            elif 'w2' in weight_name:
                init_mat = init_weight_matrix1
            else:
                raise NotImplementedError
            if 'T' in weight_name:
                init_mat = init_mat.T
            if 'm' in weight_name:
                init_mat = -init_mat

            self.loihi_connections[plastic_connection_map[weight_name]].setSynapseState('weight', init_mat.flatten())
            conn_state = self.loihi_connections[plastic_connection_map[weight_name]].getConnectionState('weight')

            if not (conn_state == init_mat).all():
               warnings.warn('rounding error in weight init on chip!')

        self.loihi_connections_plastic = [conn for conn in self.loihi_connections if
                                          (conn.endswith("_p") or conn.endswith("_f"))]
        # self.loihi_connections[conn].getConnectionState('weight')

        print('weights initialized')

        if self.do_train:
            proto_reinf = nx.ConnectionPrototype()

            use_reward_spikegen = True
            # We can do this with a neuron, but only one neuron is allowed to
            # connect to a reinforcement channel. I.e. a neuron has to be gated on by the correct gating chain neurons,
            # a direct connection from the gating chain doesn't work
            # If we use one neuron however, there are too many axons from that neuron.
            # Maybe it is possible to arrange the post neurons in a smart way to save axons? But for now, use spikegen
            if use_reward_spikegen:
                self.reward_times_per_phase = np.asarray([6, 8])

                phase_times = (np.sort((np.arange(len(input_data)) * num_gate))).tolist()
                # times_reward = np.concatenate([t + self.reward_times_per_phase for t in phase_times]).tolist()

                spike_gen_reward = net.createSpikeGenProcess(1)
                self.loihi_groups['reward'] = spike_gen_reward

                # self.loihi_groups['reward'].addSpikes(spikeInputPortNodeIds=0, spikeTimes=times_reward)

            else:
                raise NotImplementedError

            for conn in self.loihi_connections_plastic:
                lr = self.loihi_connections[conn].nodeSet._learningRules[0]

                if self.verbose:
                    print(conn)
                    print(lr)
                    print(lr.reinforcementChannel)
                try:
                    if use_reward_spikegen:
                        self.loihi_reinf_connections[conn + "_reinf"] = spike_gen_reward.connect(
                            lr.reinforcementChannel)
                    else:
                        self.loihi_reinf_connections[conn + "_reinf"] = self.loihi_groups['rew'][0].connect(
                            lr.reinforcementChannel)
                except NotImplementedError:
                    pass  # This is ok, since there is probably only one reinforcement channel,
                    # but this will try to connect to it several times, which doesn't work

        add_spikes_to_spikegen(self.loihi_groups['in_g'], indices=[0],
                               spiketimes=np.asarray([params["num_gate"] - 1]),
                               verbose=self.verbose)

        self.spikes = None
        self.weights = {}
        self.w_final = {}
        self.num_gate = num_gate
        self.num_trials = num_trials
        self.num_steps = num_steps

        self.monitor_layers = []
        self.monitor_weights = []

        self.weightprobes = {}
        self.spikeprobes = {}
        self.probe_mode = None
        self.current_offset = 0
        self.timestamp = time.strftime("%Y%m%d_%H%M")

    def setup_probes(self, probe_mode=1):
        print('setting up probes...')

        num_gate = self.num_gate
        num_trials = self.num_trials
        num_steps = self.num_steps
        self.probe_mode = probe_mode

        if probe_mode == 0:
            # no probes (power or time measurement)
            weight_dt = 2 ** 22
            weight_tstart = 2 ** 22
            spike_tstart = 2 ** 22
        elif probe_mode == 1:
            # probe only weight (loop) every 30000 (for training)
            self.monitor_weights = [w for w in self.loihi_connections_plastic if ('_p' in w) and not ('copy' in w)]
            weight_dt = num_gate * 30000
            weight_tstart = num_gate * 30000 - 1
            spike_tstart = 2 ** 22
        elif probe_mode == 2:
            # probe <100 trials with all layers and all weights (debug)
            self.monitor_weights = [w for w in self.loihi_connections_plastic if ('_p' in w) and not ('copy' in w)]
            self.monitor_layers = ['x',
                                   'h1', 'h1_copy', 'h1_copy2',
                                   'o', 'o_copy', 'o_copy2',
                                   't', 'd+', 'd-',
                                   'oT-', 'h1T', 'c_h1']
            probe_time = 101
            spike_tstart = max(num_gate * (num_trials - probe_time) + 1, 1)
            weight_dt = num_trials * num_gate
            weight_tstart = num_gate * num_trials - 1
        elif probe_mode == 3:
            # probe only output spikes for all trials (test set run)
            self.monitor_layers = ['o']
            spike_tstart = 1
            weight_dt = 2 ** 22
            weight_tstart = 2 ** 22
        else:
            raise NotImplementedError

        if not probe_mode == 0:
            sPc = [nx.SpikeProbeCondition(dt=1, tStart=spike_tstart)]
            wPc = [nx.IntervalProbeCondition(dt=weight_dt, tStart=weight_tstart)]
            # vPc = [nx.IntervalProbeCondition(dt=1, tStart=weight_tstart)]

            PP = nx.ProbeParameter
            pp_s = [PP.SPIKE]
            pp_w = [PP.SYNAPSE_WEIGHT]
            # pp_v = [PP.COMPARTMENT_VOLTAGE]

            for lay in self.monitor_layers:
                self.spikeprobes[lay] = self.loihi_groups[lay].probe(pp_s, probeConditions=sPc)[0]

                # self.vprobes = {}
                # if num_trials < 50:
                #     for lay in ['h1', 'h1_copy', 'o', 'o_copy']:
                #         self.vprobes[lay] = self.loihi_groups[lay].probe(pp_v, probeConditions=vPc)[0]

            for conn in self.monitor_weights:
                if self.verbose:
                    print(conn)
                self.weightprobes[conn] = self.loihi_connections[conn].probe(pp_w, probeConditions=wPc)

        print('done.')

    def print_save_eprobe(self, board, e_probe, params):
        # e_probe.plotEnergy()
        # e_probe.plotExecutionTime()
        print(board.energyTimeMonitor.powerProfileStats)

        print(e_probe.totalTimePerTimeStep)

        print('totalEnergy', e_probe.totalEnergy)
        total_energy = e_probe.totalEnergy - e_probe.totalHostPhaseEnergy

        print('totalHostPhaseEnergy', e_probe.totalHostPhaseEnergy, e_probe.totalHostPhaseEnergy / total_energy)
        # sum of energy for host phase for ALL timesteps
        print('totalLearningPhaseEnergy', e_probe.totalLearningPhaseEnergy,
              e_probe.totalLearningPhaseEnergy / total_energy)
        # sum of energy for learning phase for ALL timesteps
        print('totalManagementPhaseEnergy', e_probe.totalManagementPhaseEnergy,
              e_probe.totalManagementPhaseEnergy / total_energy)
        # sum of energy for management phase for ALL timesteps
        print('totalPreLearnManagementPhaseEnergy', e_probe.totalPreLearnManagementPhaseEnergy,
              e_probe.totalPreLearnManagementPhaseEnergy / total_energy)
        # sum of energy for preLearn management phase for ALL timesteps
        print('totalSpikingPhaseEnergy', e_probe.totalSpikingPhaseEnergy,
              e_probe.totalSpikingPhaseEnergy / total_energy)
        # sum of energy for spiking phase for ALL timesteps

        energy_timestep_results = {}
        energy_timestep_results['spikingPhaseEnergyPerTimeStep'] = e_probe.spikingPhaseEnergyPerTimeStep
        energy_timestep_results['managementPhaseEnergyPerTimeStep'] = e_probe.managementPhaseEnergyPerTimeStep
        energy_timestep_results['learningPhaseEnergyPerTimeStep'] = e_probe.learningPhaseEnergyPerTimeStep
        energy_timestep_results[
            'preLearnManagementPhaseEnergyPerTimeStep'] = e_probe.preLearnManagementPhaseEnergyPerTimeStep
        energy_timestep_results['hostPhaseEnergyPerTimeStep'] = e_probe.hostPhaseEnergyPerTimeStep

        energy_timestep_results['spikingTimePerTimeStep'] = e_probe.spikingTimePerTimeStep
        energy_timestep_results['managementTimePerTimeStep'] = e_probe.managementTimePerTimeStep
        energy_timestep_results['learningTimePerTimeStep'] = e_probe.learningTimePerTimeStep
        energy_timestep_results['preLearningMgmtTimePerTimeStep'] = e_probe.preLearningMgmtTimePerTimeStep
        energy_timestep_results['hostTimePerTimeStep'] = e_probe.hostTimePerTimeStep

        energy_timestep_results['totalHostPhaseEnergy'] = e_probe.totalHostPhaseEnergy
        # sum of energy for host phase for ALL timesteps
        energy_timestep_results['totalLearningPhaseEnergy'] = e_probe.totalLearningPhaseEnergy
        # sum of energy for learning phase for ALL timesteps
        energy_timestep_results['totalManagementPhaseEnergy'] = e_probe.totalManagementPhaseEnergy
        # sum of energy for management phase for ALL timesteps
        energy_timestep_results['totalPreLearnManagementPhaseEnergy'] = e_probe.totalPreLearnManagementPhaseEnergy
        # sum of energy for preLearn management phase for ALL timesteps
        energy_timestep_results['totalSpikingPhaseEnergy'] = e_probe.totalSpikingPhaseEnergy
        # sum of energy for spiking phase for ALL timesteps

        energy_timestep_results['powerProfileStats'] = board.energyTimeMonitor.powerProfileStats

        energy_timestep_results['powerProfileStats']['numtrials'] = params['num_trials']
        energy_timestep_results['powerProfileStats']['timestepspertrial'] = params['num_gate']
        energy_timestep_results['parameters'] = params

        str_time = time.strftime("_%H%M")
        filename_p = os.path.join(".", "saved_energy", "power_stats_" + self.timestamp + str_time +
                                  '_s' + str(self.params['seed']) +
                                  '_hid' + str(params['num_populations']['hid']) + ".npz")
        np.savez(filename_p, **energy_timestep_results)

    def generate_new_input_data(self, num_trials=60000):
        self.input_data, self.target_data = generate_input_data(num_trials, input_data=self.dataset, add_bias=False)
        return self.input_data, self.target_data

    def add_spikes_to_spikegen(self, chunksize=30000, full_size=60000):
        offset_trials = self.current_offset // self.num_gate

        print('adding input data from', (offset_trials % full_size), 'to', (offset_trials % full_size + chunksize),
              'with offset', offset_trials)

        indices_in, times_in = calc_spiketimes_from_input_arr(
            self.input_data[(offset_trials % full_size):(offset_trials % full_size + chunksize)],
            interval=self.num_gate,
            max_rate=1, num_neurons=1, T=1)
        add_spikes_to_spikegen(self.loihi_groups['input'], indices=indices_in,
                               spiketimes=np.asarray(times_in) + self.params[
                                   "num_gate"] + self.current_offset,
                               verbose=self.verbose)
        if self.do_train:
            indices_out, times_out = calc_spiketimes_from_input_arr(
                self.target_data[(offset_trials % full_size):(offset_trials % full_size + chunksize)],
                interval=self.num_gate,
                max_rate=1, num_neurons=1, T=1)
            add_spikes_to_spikegen(self.loihi_groups['in_tgt'], indices=indices_out,
                                   spiketimes=np.asarray(times_out) + 2 + self.num_gate + self.current_offset,
                                   verbose=self.verbose)

            phase_times = (np.sort((np.arange(chunksize) * self.num_gate) + self.current_offset)).tolist()
            times_reward = np.concatenate([t + self.reward_times_per_phase for t in phase_times]).tolist()
            self.loihi_groups['reward'].addSpikes(spikeInputPortNodeIds=0, spikeTimes=times_reward)

        start_time_compile = time.time()
        self.net.compiler.compiler.recompileProcesses()
        end_time_compile = time.time()
        print('recompiled processes for', end_time_compile - start_time_compile)

    def run(self):
        print('compiling...')
        num_trials = self.params['num_trials']
        num_steps = self.params['num_steps']

        start_time_compile = time.time()
        board = self.net.compiler.board
        if not board:
            board = self.net.compiler.compile(self.net)
        end_time_compile = time.time()
        print('compiled for', end_time_compile - start_time_compile)

        if self.do_probe_energy:
            e_probe = board.probe(nx.ProbeParameter.ENERGY,
                                  nx.PerformanceProbeCondition(tStart=1, tEnd=int(num_steps),
                                                               bufferSize=512, binSize=512))

        datasize = 60000
        print('running...')
        if num_trials < 60000:
            self.add_spikes_to_spikegen(chunksize=num_steps // self.num_gate)
            board.run(int(num_steps))
        else:
            self.add_spikes_to_spikegen(chunksize=datasize, full_size=datasize)
            num_parts = int(np.ceil(num_trials / 30000))
            for part in range(num_parts):
                if part == (num_parts - 1):
                    board.run(num_steps - 30000 * self.num_gate * (num_parts - 1))
                    self.current_offset += num_steps - 30000 * self.num_gate * (num_parts - 1)
                else:
                    board.run(30000 * self.num_gate)
                    self.current_offset += 30000 * self.num_gate

                    print('part done: ', part + 1, '/', str(num_parts))
                    # self.calc_weights()
                    self.save_results()
                    self.accuracy_from_weights()

                    if (self.current_offset % (datasize * self.num_gate)) == 0:
                        self.generate_new_input_data(num_trials=datasize)
                        self.add_spikes_to_spikegen(chunksize=datasize, full_size=datasize)

        board.disconnect()
        print('board disconnected')

        if self.do_probe_energy:
            self.print_save_eprobe(board, e_probe, self.params)

            # e_probe.plotPower()
            if self.on_kapohobay:
                self.telemetry_server_process.terminate()
                # process.wait()

            # stop
            # sys.exit()

    def calc_weights(self):
        size_hid = self.params['num_populations']['hid']
        size_in = self.params['num_populations']['in']
        size_out = self.params['num_populations']['out']

        for wgt in self.plastic_connection_map:
            try:
                print(wgt)
                probe = self.weightprobes[self.plastic_connection_map[wgt]]
                probedata = np.zeros((len(probe), len(probe[0][0].data)))
                for i, p in enumerate(probe):
                    probedata[i] = p[0].data

                w = probedata
                if 'w1' in wgt:
                    self.weights[wgt] = w.reshape((size_hid, size_in, -1))
                elif 'w2T' in wgt:
                    self.weights[wgt] = w.reshape((size_hid, size_out, -1))
                elif 'w2' in wgt:
                    self.weights[wgt] = w.reshape((size_out, size_hid, -1))
                else:
                    raise NotImplementedError

                self.w_final[wgt] = self.weights[wgt][:, :, -1]
            except KeyError as e:
                print(e)

    def save_results(self):
        self.calc_weights()
        w_final = self.w_final

        str_time = time.strftime("_%H%M")

        if self.do_train:
            filename_w = os.path.join(".", "saved_weights", "final_weights_" + self.timestamp + str_time + '_s' +
                                      str(self.params['seed']) + '_hid' +
                                      str(self.params['num_populations']['hid']) + ".npz")
            np.savez(filename_w, **w_final)

            if self.weights['w2'].shape[2] > 1:
                np.savez(filename_w.replace('final_weights', 'all_weights'), **self.weights)

            print('weights saved as', filename_w)
        else:
            for weight_name in w_final:
                conn_state = self.loihi_connections[self.plastic_connection_map[weight_name]].getConnectionState(
                    'weight')
                assert (conn_state == w_final[weight_name]).all()  # This is actually just the init matrix

        filename_s = os.path.join(".", "saved_spikes", "spikes_" + self.timestamp + str_time + ".npz")
        # filename_s = os.path.join(".", "saved_spikes",'spikes_20210409_1705.npz')

        print(self.spikeprobes)
        spikes = {sp: self.spikeprobes[sp].data for sp in self.spikeprobes}
        if len(spikes) > 0 and self.probe_mode in [2,3]:
            # spike_tstart = max(self.num_gate * (self.num_trials - 101) + 1, 1)
            # t and x usually don't need to be monitored as they are sent to the chip
            try:
                spikes['t']
            except KeyError:
                try:
                    # target_spikes = self.target_data[(spike_tstart // self.params['num_gate']):self.params['num_trials']]
                    target_spikes = self.target_data[0:self.params['num_trials']]
                    target_times, target_indices = np.where(target_spikes)
                    spikes['t'] = np.zeros((self.params['num_populations']['out'],
                                           np.max(target_times * self.params['num_gate'])))
                    spikes['t'][target_indices, target_times * self.params['num_gate']] = 1
                except Exception as e:
                    print('could not save target spikes')
                    print(e)
            try:
                spikes['x']
            except KeyError:
                try:
                    # in_spikes = self.input_data[(spike_tstart // self.params['num_gate']):self.params['num_trials']]
                    in_spikes = self.input_data[0:self.params['num_trials']]
                    in_times, in_indices = np.where(in_spikes)
                    spikes['x'] = np.zeros((self.params['num_populations']['in'],
                                           np.max(in_times * self.params['num_gate'])))
                    spikes['x'][in_indices, in_times * self.params['num_gate']] = 1
                except Exception as e:
                    print(e)

            np.savez(filename_s, **spikes)
            print('spikes saved as', filename_s)
            self.spikes = spikes

    def load_spikes(self, filename_s):
        with np.load(filename_s) as f:
            spikes_loaded = {sp: d for sp, d in f.items()}
            self.spikes = spikes_loaded

    def accuracy_from_weights(self):
        data = self.dataset.replace('_test', '')
        data += '_test'
        print('test set:')
        input_data, target_data = generate_input_data(10000, input_data=data, add_bias=False)
        inference_from_weights(self, labels=target_data, inp=input_data)

        data = self.dataset.replace('_test', '')
        print('train set:')
        input_data, target_data = generate_input_data(60000, input_data=data, add_bias=False)
        inference_from_weights(self, labels=target_data, inp=input_data)
