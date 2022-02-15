"""
Implementation of Leaky Integrate and Fire (LIF).
"""

from PymoNNto import Behaviour
import numpy as np


class LIF(Behaviour):
    """
    The neural dynamics of LIF is defined by:

    dv/dt = v_rest - v + R.I,
    if v >= threshold then v = v_reset.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        v_rest (float): voltage at rest.
        v_reset (float): value of voltage reset.
        threshold (float): the voltage threshold.
        R (float): the resistance of the membrane potential.
    """
    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag('LIF')
        self.set_init_attrs_as_variables(neurons)
        neurons.v = neurons.get_neuron_vec() * neurons.v_rest
        neurons.spikes = neurons.get_neuron_vec() > neurons.threshold
        neurons.dt = 1.

    def new_iteration(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        dv_dt = neurons.v_rest - neurons.v + neurons.R * neurons.I
        neurons.v += dv_dt * neurons.dt / neurons.tau
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset
