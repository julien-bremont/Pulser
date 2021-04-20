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

import numpy as np
import pytest

import qutip
from copy import deepcopy

from pulser import Sequence, Pulse, Register, Simulation
from pulser.devices import Chadoq2
from pulser.waveforms import BlackmanWaveform

q_dict = {"A": np.array([0., 0.]),
          "B": np.array([0., 10.]),
          }
reg = Register(q_dict)

duration = 1000
pi = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0., 0)

seq = Sequence(reg, Chadoq2)

# Declare Channels
seq.declare_channel('ryd', 'rydberg_global')
seq.add(pi, 'ryd')
seq_no_meas = deepcopy(seq)
seq.measure('ground-rydberg')

sim = Simulation(seq)
results = sim.run(spam=True)

state = qutip.tensor([qutip.basis(2, 0), qutip.basis(2, 0)])
ground = qutip.tensor([qutip.basis(2, 1), qutip.basis(2, 1)])


def test_initialization():
    assert results._dim == 2
    assert results._size == 2
    assert results._basis_name == 'ground-rydberg'
    assert results._meas_basis == 'ground-rydberg'


def test_expect():
    with pytest.raises(TypeError, match="must be a list"):
        results.expect('bad_observable')
    with pytest.raises(TypeError, match="Incompatible type"):
        results.expect(['bad_observable'])
    with pytest.raises(ValueError, match="Incompatible shape"):
        results.expect([np.array(3)])
    op = [qutip.tensor(qutip.qeye(2),
                       qutip.basis(2, 1)*qutip.basis(2, 1).dag())]
    return results.expect(op)[0]


def test_sample_final_state():
    sim_no_meas = Simulation(seq_no_meas)
    results_no_meas = sim_no_meas.run(spam=True)
    with pytest.raises(ValueError, match="can only be"):
        results_no_meas.sample_final_state('wrong_measurement_basis')

    sampling = results.sample_final_state(N_samples=1234)
    assert results.N_samples == 1234
    assert len(sampling) == 4  # Check that all states were observed.

    seq_no_meas.declare_channel('raman', 'raman_local', 'B')
    seq_no_meas.add(pi, 'raman')
    res_3level = Simulation(seq_no_meas).run()
    sampling_three_level = res_3level.sample_final_state(meas_basis='digital')
    # Raman pi pulse on one atom will not affect other,
    # even with global pi on rydberg
    assert len(sampling_three_level) == 2
    sampling_three_levelB = res_3level.sample_final_state(
                                meas_basis='ground-rydberg')
    # Global Rydberg will affect both:
    assert len(sampling_three_levelB) == 4
