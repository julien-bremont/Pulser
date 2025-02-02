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

from pulser import Sequence, Pulse, Register, Simulation
from pulser.devices import Chadoq2
from pulser.waveforms import BlackmanWaveform

q_dict = {"control1": np.array([-4., 0.]),
          "target": np.array([0., 4.]),
          "control2": np.array([4., 0.])}
reg = Register(q_dict)

duration = 1000
pi = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0., 0)
twopi = Pulse.ConstantDetuning(BlackmanWaveform(duration, 2*np.pi), 0., 0)
pi_Y = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0., -np.pi/2)

seq = Sequence(reg, Chadoq2)

# Declare Channels
seq.declare_channel('ryd', 'rydberg_local', 'control1')
seq.declare_channel('raman', 'raman_local', 'control1')

d = 0  # Pulse Duration

# Prepare state 'hhh':
seq.add(pi_Y, 'raman')
seq.target('target', 'raman')
seq.add(pi_Y, 'raman')
seq.target('control2', 'raman')
seq.add(pi_Y, 'raman')
d += 3

prep_state = qutip.tensor([qutip.basis(3, 2) for _ in range(3)])

# Write CCZ sequence:
seq.add(pi, 'ryd', protocol='wait-for-all')
seq.target('control2', 'ryd')
seq.add(pi, 'ryd')
seq.target('target', 'ryd')
seq.add(twopi, 'ryd')
seq.target('control2', 'ryd')
seq.add(pi, 'ryd')
seq.target('control1', 'ryd')
seq.add(pi, 'ryd')
d += 5


def test_initialization_and_construction_of_hamiltonian():
    fake_sequence = {'pulse1': 'fake', 'pulse2': "fake"}
    with pytest.raises(TypeError, match='sequence has to be a valid'):
        Simulation(fake_sequence)
    sim = Simulation(seq, sampling_rate=0.011)
    assert sim._seq == seq
    assert sim._qdict == seq.qubit_info
    assert sim._size == len(seq.qubit_info)
    assert sim._tot_duration == duration * d
    assert sim._qid_index == {"control1": 0, "target": 1, "control2": 2}

    with pytest.raises(ValueError, match='too small, less than'):
        Simulation(seq, sampling_rate=0.0001)
    with pytest.raises(ValueError, match='positive and not larger'):
        Simulation(seq, sampling_rate=5)
    with pytest.raises(ValueError, match='positive and not larger'):
        Simulation(seq, sampling_rate=-1)

    assert sim.sampling_rate == 0.011
    assert len(sim._times) == int(sim.sampling_rate * sim._tot_duration)

    assert isinstance(sim._hamiltonian, qutip.QobjEvo)
    # Checks adapt() method:
    assert bool(set(sim._hamiltonian.tlist).intersection(sim._times))
    for qobjevo in sim._hamiltonian.ops:
        for sh in qobjevo.qobj.shape:
            assert sh == sim.dim**sim._size


def test_extraction_of_sequences():
    sim = Simulation(seq)
    for channel in seq.declared_channels:
        addr = seq.declared_channels[channel].addressing
        basis = seq.declared_channels[channel].basis

        if addr == 'Global':
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    samples = sim.samples[addr][basis]
                    assert (samples['amp'][slot.ti:slot.tf]
                            == slot.type.amplitude.samples).all()
                    assert (samples['det'][slot.ti:slot.tf]
                            == slot.type.detuning.samples).all()
                    assert (samples['phase'][slot.ti:slot.tf]
                            == slot.type.phase).all()

        elif addr == 'Local':
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    for qubit in slot.targets:  # TO DO: multiaddressing??
                        samples = sim.samples[addr][basis][qubit]
                        assert (samples['amp'][slot.ti:slot.tf]
                                == slot.type.amplitude.samples).all()
                        assert (samples['det'][slot.ti:slot.tf]
                                == slot.type.detuning.samples).all()
                        assert (samples['phase'][slot.ti:slot.tf]
                                == slot.type.phase).all()


def test_building_basis_and_projection_operators():
    # All three levels:
    sim = Simulation(seq, sampling_rate=0.01)
    assert sim.basis_name == 'all'
    assert sim.dim == 3
    assert sim.basis == {'r': qutip.basis(3, 0),
                         'g': qutip.basis(3, 1),
                         'h': qutip.basis(3, 2)}
    assert (sim.op_matrix['sigma_rr'] ==
            qutip.basis(3, 0) * qutip.basis(3, 0).dag())
    assert (sim.op_matrix['sigma_gr'] ==
            qutip.basis(3, 1) * qutip.basis(3, 0).dag())
    assert (sim.op_matrix['sigma_hg'] ==
            qutip.basis(3, 2) * qutip.basis(3, 1).dag())

    # Check local operator building method:
    with pytest.raises(ValueError, match="Duplicate atom"):
        sim._build_operator('sigma_gg', "target", "target")

    # Global ground-rydberg
    seq2 = Sequence(reg, Chadoq2)
    seq2.declare_channel('global', 'rydberg_global')
    seq2.add(pi, 'global')
    sim2 = Simulation(seq2, sampling_rate=0.01)
    assert sim2.basis_name == 'ground-rydberg'
    assert sim2.dim == 2
    assert sim2.basis == {'r': qutip.basis(2, 0),
                          'g': qutip.basis(2, 1)
                          }
    assert (sim2.op_matrix['sigma_rr'] ==
            qutip.basis(2, 0) * qutip.basis(2, 0).dag())
    assert (sim2.op_matrix['sigma_gr'] ==
            qutip.basis(2, 1) * qutip.basis(2, 0).dag())

    # Digital
    seq2b = Sequence(reg, Chadoq2)
    seq2b.declare_channel('local', 'raman_local', 'target')
    seq2b.add(pi, 'local')
    sim2b = Simulation(seq2b, sampling_rate=0.01)
    assert sim2b.basis_name == 'digital'
    assert sim2b.dim == 2
    assert sim2b.basis == {'g': qutip.basis(2, 0),
                           'h': qutip.basis(2, 1)
                           }
    assert (sim2b.op_matrix['sigma_gg'] ==
            qutip.basis(2, 0) * qutip.basis(2, 0).dag())
    assert (sim2b.op_matrix['sigma_hg'] ==
            qutip.basis(2, 1) * qutip.basis(2, 0).dag())

    # Local ground-rydberg
    seq2c = Sequence(reg, Chadoq2)
    seq2c.declare_channel('local_ryd', 'rydberg_local', 'target')
    seq2c.add(pi, 'local_ryd')
    sim2c = Simulation(seq2c, sampling_rate=0.01)
    assert sim2c.basis_name == 'ground-rydberg'
    assert sim2c.dim == 2
    assert sim2c.basis == {'r': qutip.basis(2, 0),
                           'g': qutip.basis(2, 1)
                           }
    assert (sim2c.op_matrix['sigma_rr'] ==
            qutip.basis(2, 0) * qutip.basis(2, 0).dag())
    assert (sim2c.op_matrix['sigma_gr'] ==
            qutip.basis(2, 1) * qutip.basis(2, 0).dag())


def test_empty_sequences():
    seq = Sequence(reg, Chadoq2)
    with pytest.raises(ValueError, match='no declared channels'):
        Simulation(seq)
    with pytest.raises(ValueError, match='No instructions given'):
        seq.declare_channel('test', 'rydberg_local', 'target')
        seq.declare_channel("test2", "rydberg_global")
        Simulation(seq)


def test_run():
    sim = Simulation(seq, sampling_rate=0.01)
    bad_initial = np.array([1.])
    good_initial_array = np.r_[1, np.zeros(sim.dim**sim._size - 1)]
    good_initial_qobj = qutip.tensor([qutip.basis(sim.dim, 0)
                                      for _ in range(sim._size)])

    with pytest.raises(ValueError,
                       match='Incompatible shape of initial_state'):
        sim.run(bad_initial)
    with pytest.raises(ValueError,
                       match='Incompatible shape of initial_state'):
        sim.run(qutip.Qobj(bad_initial))

    sim.run(initial_state=good_initial_array)
    sim.run(initial_state=good_initial_qobj)

    assert not hasattr(sim._seq, '_measurement')
    seq.measure('ground-rydberg')
    sim.run()
    assert sim._seq._measurement == 'ground-rydberg'
