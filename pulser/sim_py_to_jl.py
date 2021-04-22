# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import qutip
import numpy as np

from copy import deepcopy
from julia import QuantumOptics as qo
from julia import Main, Base
from pulser import Pulse, Sequence
from pulser.simresults import CleanResults
from collections import namedtuple
from scipy.interpolate import interp1d

_TimeSlot = namedtuple('_TimeSlot', ['type', 'ti', 'tf', 'targets'])

Main.include("Pulser\\pulser\\qo_jl_sim.jl")


class Simulation:
    """Simulation of a pulse sequence using QuTiP.

    Creates a Hamiltonian object with the proper dimension according to the
    pulse sequence given, then provides a method to time-evolve an initial
    state using the QuTiP solvers.

    Args:
        sequence (Sequence): An instance of a Pulser Sequence that we
            want to simulate.

    Keyword Args:
        sampling_rate (float): The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0
    """

    def __init__(self, sequence, sampling_rate=1.0, noise={"Doppler": False},
                 damping={"Emission": False}):
        """Initialize the Simulation with a specific pulser.Sequence."""
        if not isinstance(sequence, Sequence):
            raise TypeError("The provided sequence has to be a valid "
                            "pulser.Sequence instance.")
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence._channels):
            raise ValueError("No instructions given for the channels in the "
                             "sequence.")
        self._seq = sequence
        self._noise = noise
        self._damping = damping
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._tot_duration = max(
            [self._seq._last(ch).tf for ch in self._seq._schedule]
        )

        if not (0 < sampling_rate <= 1.0):
            raise ValueError("`sampling_rate` must be positive and "
                             "not larger than 1.0")
        if int(self._tot_duration*sampling_rate) < 4:
            raise ValueError("`sampling_rate` is too small, less than 4 data "
                             "points.")
        self.sampling_rate = sampling_rate

        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)

        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

    def _extract_samples(self):
        """Populate samples dictionary with every pulse in the sequence."""

        def prepare_dict():
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot, samples_dict):
            samples_dict['amp'][slot.ti:slot.tf] += slot.type.amplitude.samples
            if(self._noise["Doppler"]):
                # sigma = k_eff \Delta v : See Sylvain's paper
                noise = 1 + np.random.normal(
                            0, 2*np.pi*0.12, slot.tf - slot.ti)
            else:
                noise = 1
            samples_dict['det'][slot.ti:slot.tf] += \
                slot.type.detuning.samples * noise
            samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            samples_dict = self.samples[addr][basis]

            if addr == 'Global':
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict)

            elif addr == 'Local':
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            write_samples(slot, samples_dict[qubit])

            self.samples[addr][basis] = samples_dict

    def _build_basis_and_op_matrices(self):
        """Determine dimension, basis and projector operators."""

        # No samples => Empty dict entry => False
        if (not self.samples['Global']['digital']
                and not self.samples['Local']['digital']):
            self.basis_name = 'ground-rydberg'
            self.dim = 2
            self.jl_basis = qo.NLevelBasis(2)
            basis = ['r', 'g']
            basis_index = {'r': 1, 'g': 2}
            projectors = ['gr', 'rr', 'gg']
        elif (not self.samples['Global']['ground-rydberg']
                and not self.samples['Local']['ground-rydberg']):
            self.basis_name = 'digital'
            self.dim = 2
            self.jl_basis = qo.NLevelBasis(2)
            basis = ['g', 'h']
            basis_index = {'g': 1, 'h': 2}
            projectors = ['hg', 'hh', 'gg']
        else:
            self.basis_name = 'all'  # All three states
            self.dim = 3
            self.jl_basis = qo.NLevelBasis(3)
            basis = ['r', 'g', 'h']
            basis_index = {'r': 1, 'g': 2, 'h': 3}
            projectors = ['gr', 'hg', 'rr', 'gg', 'hh', 'hr']
        self.tensor_jl_basis = Main.tensor_basis(self.jl_basis, self._size)
        # Julia arrays start at 1
        self.basis = {b: qo.nlevelstate(self.jl_basis, i+1) for
                      i, b in enumerate(basis)}
        self.op_matrix = {'I': qo.identityoperator(self.jl_basis)}

        for proj in projectors:
            self.op_matrix['sigma_' + proj] = (
                qo.transition(self.jl_basis, basis_index[proj[0]],
                              basis_index[proj[1]])
            )

    def _build_operator(self, op_id, *qubit_ids, global_op=False):
        """Create qo.jl operator with non trivial action at qubit_ids."""
        # REGLER LE PB POUR N = 1
        qindex = [self._qid_index.get(id)+1 for id in qubit_ids]
        op_list = [self.op_matrix[op_id] for _ in range(len(qindex))]
        print(qindex)
        return Main.build_operator(op_list, self.tensor_jl_basis,
                                   qindex, global_op)

    def _construct_hamiltonian(self):
        def adapt(full_array):
            """Adapt list to correspond to sampling rate"""
            indexes = np.linspace(0, self._tot_duration-1,
                                  int(self.sampling_rate*self._tot_duration),
                                  dtype=int)
            return full_array[indexes]

        def make_vdw_term():
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = Main.product_list([qo.identityoperator(self.tensor_jl_basis), 0])
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                dist = np.linalg.norm(
                    self._qdict[q1] - self._qdict[q2])
                U = 0.5 * self._seq._device.interaction_coeff / dist**6
                # + doesn't work...
                vdw = Base.sum([vdw,
                               Main.product_list([U,
                                                 self._build_operator(
                                                 'sigma_rr',
                                                 q1, q2)])])
            return vdw

        def build_coeffs_ops(basis, addr):
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']

            terms = []
            if addr == 'Global':
                coeffs = [0.5*samples['amp'] * np.exp(-1j * samples['phase']),
                          -0.5 * samples['det']]
                for op_id, coeff in zip(op_ids, coeffs):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] =\
                                self._build_operator(op_id, global_op=True)
                        terms.append([operators[op_id], adapt(coeff)])
            elif addr == 'Local':
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [0.5*samples_q['amp'] *
                              np.exp(-1j * samples_q['phase']),
                              -0.5 * samples_q['det']]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = \
                                    self._build_operator(op_id, q_id)
                            terms.append([operators[q_id][op_id],
                                          adapt(coeff)])

            self.operators[addr][basis] = operators
            return terms

        def _interpolate_coeffs(op_coef_list):
            # op_coef_list = terms
            """Each coefficient is now a function interpolated from the fixed
                values already given"""
            times = self._times
            terms_interp = []
            print(op_coef_list[0])
            for [op, coef] in op_coef_list:
                interp_coef = interp1d(times, coef)
                print(type(interp_coef))
                terms_interp.append((op, interp_coef))
            return terms_interp

        def _build_hamiltonian(terms):
            def f(t):
                h = Main.product_list([0, qo.identityoperator(self.tensor_jl_basis)])
                for (o, c) in terms:
                    coef = np.real(c(t))
                    print(coef * o)
                    h = Base.sum(h, Main.product_list([coef, o]))
                h = Base.sum(h, qo.dagger(h))
                return h
            return f

        # Time independent term:
        if self.basis_name == 'digital':
            terms = []
        else:
            # Van der Waals Interaction Terms
            # terms = [make_vdw_term()] if self._size > 1 else []
            terms = []
        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    terms += build_coeffs_ops(basis, addr)

        self._times = adapt(np.arange(self._tot_duration,
                                      dtype=np.double)/1000)

        interp_terms = _interpolate_coeffs(terms)

        f = _build_hamiltonian(interp_terms)
        print(f(0))
        self._hamiltonian = f

    def get_hamiltonian(self, time):
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time (float): The specific time in which we want to extract the
                    Hamiltonian (in ns).

        Returns:
            Qutip.Qobj: A new Qobj for the Hamiltonian with coefficients
                    extracted from the effective sequence (determined by
                    `self.sampling_rate`) at the specified time.
        """
        if time > 1000 * self._times[-1]:
            raise ValueError("Provided time is larger than sequence duration.")
        if time < 0:
            raise ValueError("Provided time is negative.")
        return self._hamiltonian(time/1000)  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, progress_bar=None, spam=False, t=-1,
            meas_basis="ground-rydberg",
            spam_dict={"eta": 0.005, "epsilon": 0.01, "epsilon_prime": 0.05},
            **options):
        """Simulate the sequence using QuTiP's solvers.

        Keyword Args:
            initial_state (array): The initial quantum state of the
               evolution. Will be transformed into a
               qutip.Qobj instance.
            progress_bar (bool): If True, the progress bar of QuTiP's sesolve()
                will be shown.
            spam (bool): If True, returns a NoisyResults object instead of a
                CleanResults one, taking into account SPAM errors.
            t (int): Time at which the results are to be returned ;
                only used with noisy simulations.
            meas_basis: Measurement basis : used with noisy simulations to
                convert a ket in Hilbert space into a bitstring (for digital
                basis).
            spam_dict: A dictionary containing SPAM error probabilities.

        Returns:
            SimulationResults: Object containing the time evolution results.
                Is a CleanResults object if spam = False, and a NoisyResults
                one if spam = True.
        """
        psi0 = Main.tensor_list([self.basis['g'] for _ in range(self._size)])
        tout, psit = qo.timeevolution.schroedinger_dynamic(
            self._times, psi0, self._hamiltonian)
        return psit
