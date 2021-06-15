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
"""Classes for containing and processing the results of a simulation."""

from __future__ import annotations
from collections import Counter
from abc import ABC, abstractmethod
from typing import Optional, Union, cast, Tuple
from collections.abc import Sequence

import matplotlib.pyplot as plt
import qutip
from qutip.piqs import isdiagonal
import numpy as np
from numpy.typing import ArrayLike


class SimulationResults(ABC):
    """Results of a simulation run of a pulse sequence.

    Parent class for NoisyResults and CleanResults.
    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, dim: int, size: int, basis_name: str,
                 sim_times: np.ndarray) -> None:
        """Initializes a new SimulationResults instance.

        Args:
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times (array): Array of times (µs) when simulation results are
                returned.
        """
        self._dim = dim
        self._size = size
        if basis_name not in {'ground-rydberg', 'digital', 'all'}:
            raise ValueError(
                "`basis_name` must be 'ground-rydberg', 'digital' or 'all'."
                )
        self._basis_name = basis_name
        self.sim_times = sim_times
        self._results: Union[list[Counter], list[qutip.Qobj]]

    @abstractmethod
    def get_state(self, t: float) -> qutip.Qobj:
        """Returns the state of the system at time t."""
        pass

    @abstractmethod
    def get_final_state(self) -> qutip.Qobj:
        """Returns the final state of the system."""
        pass

    @abstractmethod
    def plot(self, op: qutip.Qobj, fmt: str = '.', label: str = '') -> None:
        """Plots the expectation value of a given operator op."""
        pass

    @abstractmethod
    def _calc_weights(self, t: float) -> ArrayLike:
        """Computes the bitstring probabilities for sampled states."""
        pass

    def expect(self, obs_list: Sequence[Union[qutip.Qobj, ArrayLike]]
               ) -> list[Union[float, complex, ArrayLike]]:
        """Returns the expectation values of operators in obs_list."""
        states = [self.get_state(t) for t in self.sim_times]
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators.")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (2**self._size, 2**self._size):
                raise ValueError("Incompatible shape of observable.")
            qobj_list.append(qutip.Qobj(obs))

        return cast(list, qutip.expect(qobj_list, states))

    def sample_state(self, t: float, N_samples: int = 1000) -> Counter:
        """Returns the result of multiple measurements at time t.

        Args:
            t (float): Time at which the state is sampled.
            N_samples (int): Number of samples to return.
        """
        dist = np.random.multinomial(N_samples, self._calc_weights(t))
        return Counter({np.binary_repr(
                        i, self._size): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, N_samples: int = 1000) -> Counter:
        """Returns the result of multiple measurements of the final state."""
        return self.sample_state(self.sim_times[-1], N_samples)

    def _get_index_from_time(self, t_float: float) -> int:
        try:
            return int(np.where(self.sim_times == t_float)[0][0])
        except IndexError:
            print(f"Given time {t_float} is absent from Simulation times.")


class NoisyResults(SimulationResults):
    """Results of a noisy simulation run of a pulse sequence.

    Contrary to a CleanResults object, this object contains a list of Counter
    describing the state distribution at the time it was created by using
    Simulation.run() with a noisy simulation.
    Contains methods for studying the populations and extracting useful
    information from them.
    """

    def __init__(self, run_output: list[Counter],
                 size: int, basis_name: str,
                 sim_times: np.ndarray, N_measures: int, dim: int = 2) -> None:
        """Initializes a new NoisyResults instance.

        Warning:
            Can't have single-atom Hilbert spaces with dimension bigger
            than 2 for NoisyResults objects.
            This is not the case for a CleanResults object, containing states
            in Hilbert space, but NoisyResults contains a probability
            distribution of bitstrings, not atomic states

        Args:
            run_output (list[Counter]): Each Counter contains the
                probability distribution of a multi-qubits state,
                represented as a bitstring. There is one Counter for each time
                the simulation was asked to return a result.
            size (int): The number of atoms in the register.
            basis_name (str): Basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg' or 'digital' - 'all' basis
                makes no sense after projection on bitstrings).
            sim_times (list): times at which Simulation object returned the
                results.
            meas_basis (Optional[str]): The basis in which a sampling
                measurement is desired.
            N_measures (int): number of measurements needed to compute this
                result when doing the simulation.
            dim (int): equals to 2 here, since projections already happened.
        """
        if basis_name == 'all':
            raise ValueError("`basis_name` must be either 'ground-rydberg' or"
                             + " 'digital'.")
        super().__init__(dim, size, basis_name, sim_times)
        self.N_measures = N_measures
        self._results = run_output

    @property
    def states(self) -> list[Counter]:
        """Probability distribution of the bitstrings."""
        return self._results

    def get_state(self, t: float) -> qutip.Qobj:
        """Get the state at time t as a diagonal density matrix.

        Note:
            This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Args:
            t (int): index of the state to be returned.

        Returns:
            qutip.Qobj: States probability distribution as a diagonal
                density matrix.
        """
        def _proj_from_bitstring(bitstring: str) -> qutip.Qobj:
            # In the digital case, |h> = |1> = qutip.basis()
            if self._basis_name == 'digital':
                proj = qutip.tensor([qutip.basis(2, int(i)).proj() for i
                                     in bitstring])
            # ground-rydberg basis case
            else:
                proj = qutip.tensor([qutip.basis(2, 1-int(i)).proj() for i
                                     in bitstring])
            return proj

        t_index = self._get_index_from_time(t)
        return sum(v * _proj_from_bitstring(b) for
                   b, v in self._results[t_index].items())

    def get_final_state(self) -> qutip.Qobj:
        """Get the final state of the simulation as a diagonal density matrix.

        Note: This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Returns:
            qutip.Qobj: States probability distribution as a density matrix.
        """
        return self.get_state(self.sim_times[-1])

    def expect(self, obs_list: Sequence[Union[qutip.Qobj, ArrayLike]]
               ) -> list[Union[float, complex, ArrayLike]]:
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.

        Note: This only works for diagonal observables, since results have been
            projected onto the Z basis.

        Returns:
            list: the list of expectation values of each operator.
        """
        for obs in obs_list:
            if not isdiagonal(obs):
                raise ValueError(f"Observable {obs} is non-diagonal.")

        return super().expect(obs_list)

    def _calc_weights(self, t: float) -> ArrayLike:
        N = self._size
        bitstrings = [np.binary_repr(k, N) for k in range(2**N)]
        t_index = self._get_index_from_time(t)
        return [self._results[t_index][b] for b in bitstrings]

    def _standard_dev(self, op: qutip.Qobj) -> ArrayLike:
        """Returns the square root of the variance of operator op."""
        density_mats = [self.get_state(t) for t in self.sim_times]
        return cast(ArrayLike,
                    np.sqrt(qutip.variance(op, density_mats) / self.N_measures)
                    )

    def _get_error_bars(self, op: qutip.Qobj) -> Tuple[ArrayLike, ArrayLike]:
        moy = self.expect([op])[0]
        st = self._standard_dev(op)
        return moy, st

    def plot(self, op: qutip.Qobj, fmt: str = '.',
             label: str = '', error_bars: bool = True) -> None:
        """Plots the expectation value of a given operator op.

        Note: The observable must be diagonal.

        Args:
            op (qutip.Qobj): Operator whose expectation value is wanted.
            error_bars (bool): Choose to display error bars.
            fmt (str): Curve plot format.
            label (str): y-Axis label.
        """
        if not isdiagonal(op):
            raise ValueError(f"Observable {op} is non-diagonal.")
        if error_bars:
            moy, st = self._get_error_bars(op)
            plt.errorbar(self.sim_times, moy, st, fmt=fmt, lw=1, capsize=3,
                         label=label)
        else:
            plt.plot(self.sim_times, self.expect([op])[0], fmt,
                     label=label)
        plt.xlabel('Time (µs)')
        plt.ylabel('Expectation value')


class CleanResults(SimulationResults):
    """Results of an ideal simulation run of a pulse sequence.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output: list[qutip.Qobj],
                 dim: int, size: int, basis_name: str,
                 sim_times: np.ndarray, meas_basis: str) -> None:
        """Initializes a new CleanResults instance.

        Args:
            run_output (list of qutip.Qobj): List of `qutip.Qobj` corresponding
                to the states at each time step after the evolution has been
                simulated.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times (list): Times at which Simulation object returned the
                results.
            meas_basis (str): The basis in which a sampling measurement
                is desired.
        """
        super().__init__(dim, size, basis_name, sim_times)
        if meas_basis:
            if meas_basis not in {'ground-rydberg', 'digital'}:
                raise ValueError(
                    "`meas_basis` must be 'ground-rydberg' or 'digital'."
                    )
        self._meas_basis = meas_basis
        self._results = run_output

    @property
    def states(self) -> list[qutip.Qobj]:
        """List of ``qutip.Qobj`` for each state in the simulation."""
        return list(self._results)

    def get_state(self, t: float, reduce_to_basis: Optional[str] = None,
                  ignore_global_phase: bool = True, tol: float = 1e-6,
                  normalize: bool = True) -> qutip.Qobj:
        """Get the state at time t of the simulation.

        Args:
            t (float): Time at which to return the state.
            reduce_to_basis (str, default=None): Reduces the full state vector
                to the given basis ("ground-rydberg" or "digital"), if the
                population of the states to be ignored is negligible.
            ignore_global_phase (bool, default=True): If True, changes the
                final state's global phase such that the largest term (in
                absolute value) is real.
            tol (float, default=1e-6): Maximum allowed population of each
                eliminated state.
            normalize (bool, default=True): Whether to normalize the reduced
                state.

        Returns:
            qutip.Qobj: The resulting final state.

        Raises:
            TypeError: If trying to reduce to a basis that would eliminate
                states with significant occupation probabilites.
        """
        t_index = self._get_index_from_time(t)
        state = cast(qutip.Qobj, self._results[t_index].copy())
        if ignore_global_phase:
            full = state.full()
            global_ph = float(np.angle(full[np.argmax(np.abs(full))]))
            state *= np.exp(-1j * global_ph)
        if self._dim != 3:
            if reduce_to_basis not in [None, self._basis_name]:
                raise TypeError(f"Can't reduce a system in {self._basis_name}"
                                + f" to the {reduce_to_basis} basis.")
        elif reduce_to_basis is not None:
            if reduce_to_basis == "ground-rydberg":
                ex_state = "2"
            elif reduce_to_basis == "digital":
                ex_state = "0"
            else:
                raise ValueError("'reduce_to_basis' must be 'ground-rydberg' "
                                 + f"or 'digital', not '{reduce_to_basis}'.")
            ex_inds = [i for i in range(3**self._size) if ex_state in
                       np.base_repr(i, base=3).zfill(self._size)]
            ex_probs = np.abs(state.extract_states(ex_inds).full()) ** 2
            if not np.all(np.isclose(ex_probs, 0, atol=tol)):
                raise TypeError(
                    "Can't reduce to chosen basis because the population of a "
                    "state to eliminate is above the allowed tolerance."
                    )
            state = state.eliminate_states(ex_inds, normalize=normalize)

        return state.tidyup()

    def get_final_state(self, reduce_to_basis: Optional[str] = None,
                        ignore_global_phase: bool = True, tol: float = 1e-6,
                        normalize: bool = True) -> qutip.Qobj:
        """Returns the final state of the Simulation."""
        return self.get_state(self.sim_times[-1], reduce_to_basis,
                              ignore_global_phase, tol, normalize)

    def _calc_weights(self, t: float) -> ArrayLike:
        t_index = self._get_index_from_time(t)
        N = self._size
        state_t = cast(qutip.Qobj, self._results[t_index]).unit()
        # Case of a density matrix
        if state_t.type != "ket":
            probs = np.abs(state_t.diag())
        else:
            probs = (np.abs(state_t.full())**2).flatten()

        if self._dim == 2:
            if self._meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. N=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                weights = probs if self._meas_basis == 'digital' \
                    else probs[::-1]
            else:
                # Only 000...000 is measured
                weights = np.zeros(probs.size)
                weights[0] = 1.

        elif self._dim == 3:
            if self._meas_basis == 'ground-rydberg':
                one_state = 0       # 1 = |r>
                ex_one = slice(1, 3)
            elif self._meas_basis == 'digital':
                one_state = 2       # 1 = |h>
                ex_one = slice(0, 2)
            probs = probs.reshape([3]*N)
            weights = np.zeros(2**N)
            for dec_val in range(2**N):
                ind: list[Union[int, slice]] = []
                for v in np.binary_repr(dec_val, width=N):
                    if v == '0':
                        ind.append(ex_one)
                    else:
                        ind.append(one_state)
                # Eg: 'digital' basis : |1> = index2, |0> = index0, 1 = 0:2
                # p_11010 = sum(probs[2, 2, 0:2, 2, 0:2])
                # We sum all probabilites that correspond to measuring
                # 11010, namely hhghg, hhrhg, hhghr, hhrhr
                weights[dec_val] = np.sum(probs[tuple(ind)])
        else:
            raise NotImplementedError(
                "Cannot sample system with single-atom state vectors of "
                "dimension > 3.")
        # Takes care of numerical artefacts in case sum(weights) != 1
        weights /= sum(weights)
        return cast(ArrayLike, weights)

    def detection_from_basis_state(self, N_d: int, shot: str,
                                   spam: dict[str, float]) -> Counter:
        """Computes the distribution of states detected when detecting `shot`.

        Part of the SPAM implementation : computes measurement errors.

        Args:
            N_d (int): Number of times state has been detected.
            shot (str): Binary string of length the number of atoms of the
            simulation.
            spam (dict): Dictionnary gathering the SPAM error
            probabilities.
        """
        n_0 = shot.count('0')
        n_1 = shot.count('1')
        eps = spam['epsilon']
        eps_p = spam['epsilon_prime']
        # Verified
        prob_1_to_0 = eps_p * (1 - eps) ** n_0 * (1 - eps_p) ** (n_1 - 1)
        prob_0_to_1 = eps * (1 - eps) ** (n_0 - 1) * (1 - eps_p) ** n_1
        probs = [int(shot[i]) * prob_1_to_0 + (1 - int(shot[i]))
                 * prob_0_to_1 for i in range(len(shot))]
        probs += [1. - sum(probs)]
        shots = np.random.multinomial(N_d, probs)
        detected_dict = {shot: shots[-1]}

        for i in range(len(shot)):
            if shots[i]:
                detected_dict[shot[:i] + str(1 - int(shot[i])) + shot[i
                              + 1:]] = shots[i]
        return Counter(detected_dict)

    def sampling_with_detection_errors(self, spam: dict[str, float],
                                       t: float,
                                       N_samples: int = 1000) -> Counter:
        """Returns the distribution of states really detected.

        Doesn't take state preparation errors into account.
        Part of the SPAM implementation.

        Args:
            spam (dict): Dictionnary gathering the SPAM error
            probabilities.
            t (float): Time at which to return the samples.
            N_samples (int): Number of samples.
        """
        sampled_state = self.sample_state(t, N_samples)
        detected_sample_dict: Counter = Counter()
        for (shot, N_d) in sampled_state.items():
            dict_state = self.detection_from_basis_state(N_d, shot, spam)
            detected_sample_dict += dict_state

        return detected_sample_dict

    def plot(self, op: qutip.Qobj, fmt: str = '', label: str = '') -> None:
        """Plots the expectation value of a given operator op.

        Args:
            op (qutip.Qobj): Operator whose expectation value is wanted.
            fmt (str): Curve plot format.
            label (str): Axis label.
            error_bars (bool): False here.
        """
        plt.plot(self.sim_times, self.expect([op])[0], fmt, label=label)
        plt.xlabel('Time (µs)')
        plt.ylabel('Expectation value')
