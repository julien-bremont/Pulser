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

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Literal, get_args

import numpy as np
import qutip


NOISE_TYPES = Literal['doppler', 'amplitude', 'SPAM', 'dephasing']
MASS = 1.45e-25
KB = 1.38e-23
KEFF = 8.7


@dataclass(frozen=True)
class SimConfig:
    """Include additional parameters to simulation.

        The user chooses the settings when creating a SimConfig. Settings
        cannot be changed after instantiation of a SimConfig.

        Keyword arguments:
            noise (tuple[NOISE_TYPES]): Types of noises to be used in the
                simulation. Choose among:
                -   'dephasing': Random phase (Z) flip
                -   'doppler': Noisy doppler runs
                -   'amplitude': Noisy gaussian beam
                -   'SPAM': SPAM errors. Adds:
                    --  eta: Probability of each atom to be badly prepared
                    --  epsilon: Probability of false positives
                    --  epsilon_prime: Probability of false negatives.
            runs (int): Number of runs needed : each run draws a new random
                noise.
            samples_per_run (int): Number of samples per noisy run.
                Useful for cutting down on computing time, but unrealistic.
            temperature (float): Temperature, set in µK, of the Rydberg array.
                Also sets the standard deviation of the speed of the atoms.
            laser_waist (float): Waist of the gaussian laser in global pulses.
            solver_options (qutip.Options): Options for the qutip solver.
        """
    noise: tuple[NOISE_TYPES] = field(default_factory=tuple)
    runs: int = 15
    samples_per_run: int = 5
    temperature: float = 50.
    laser_waist: float = 175.
    eta: float = 0.005
    epsilon: float = 0.01
    epsilon_prime: float = 0.05
    solver_options: qutip.Options = qutip.Options(max_step=5)

    def __post_init__(self):
        self._process_temperature()
        self._check_noise_types()
        self.__dict__["spam_dict"] = {'eta': self.eta, 'epsilon': self.epsilon,
                                      'epsilon_prime': self.epsilon_prime}
        self._check_spam_dict()
        self._calc_sigma_doppler()

    def __str__(self) -> str:
        lines = [
            "Options:",
            "----------",
            "Noise types:         " + ", ".join(self.noise),
            f"Spam dictionary:     {self.spam_dict}",
            f"Temperature:         {self.temperature}K",
            f"Number of runs:      {self.runs}",
            f"Samples per runs:    {self.samples_per_run}",
            f"Laser waist:         {self.laser_waist}μm",
            "Solver Options:",
            f"{str(self.solver_options)[10:-1]}",
            ]
        return "\n".join(lines)

    def _check_spam_dict(self) -> None:
        for param, value in self.spam_dict.items():
            if value > 1 or value < 0:
                raise ValueError(f"SPAM parameter {param} = {value} must be"
                                 + " greater than 0 and less than 1.")

    def _process_temperature(self) -> None:
        # checks value of temperature field and converts it to K from muK
        if self.temperature <= 0:
            raise ValueError("Temperature field"
                             + f" (`temperature` = {self.temperature}) must be"
                             + " greater than 0.")
        self.__dict__["temperature"] *= 1.e-6

    def _check_noise_types(self) -> None:
        # only one noise was given as argument : convert it to a tuple
        if isinstance(self.noise, str):
            self.__dict__["noise"] = (self.noise, )
        for noise_type in self.noise:
            if noise_type not in get_args(NOISE_TYPES):
                raise ValueError(str(noise_type)+" is not a valid noise type."
                                 "Valid noise types : " + get_args(NOISE_TYPES)
                                 )

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self.__dict__["doppler_sigma"]: float = KEFF * np.sqrt(
            KB * self.temperature / MASS)
