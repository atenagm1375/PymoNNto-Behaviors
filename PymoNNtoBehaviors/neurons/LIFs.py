"""
Implementation of Leaky Integrate and Fire (LIF) and its variants.
"""

import numpy as np
from PymoNNto import Behaviour


class LIF(Behaviour):
    """
    The neural dynamics of LIF is defined by:

    tau*dv/dt = v_rest - v + R*I,
    if v >= threshold then v = v_reset.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        v_rest (float): voltage at rest.
        v_reset (float): value of voltage reset.
        threshold (float): the voltage threshold.
        R (float): the resistance of the membrane potential.
    """

    def _initialize(self, neurons):
        self.set_init_attrs_as_variables(neurons)
        neurons.v = neurons.get_neuron_vec(mode="ones()") * neurons.v_rest
        neurons.spikes = neurons.get_neuron_vec(mode="zeros()")

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("LIF")
        self._initialize(neurons)

    def _dv_dt(self, neurons):
        """
        Single step voltage dynamics of simple LIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        delta_v = neurons.v_rest - neurons.v
        ri = neurons.R * neurons.I
        return delta_v + ri

    def _fire(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

    def new_iteration(self, neurons):
        """
        Firing behavior of LIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self._fire(neurons)

        neurons.v += self._dv_dt(neurons) / neurons.tau


class ELIF(LIF):
    """
    The neural dynamics of Exponential LIF is defined by:

    F(u) = sharpness * exp((v - theta_rh) / sharpness),
    tau*dv/dt = v_rest - v + F(u) + R*I,
    if v >= threshold then v = v_reset.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        v_rest (float): voltage at rest.
        v_reset (float): value of voltage reset.
        threshold (float): the voltage threshold.
        R (float): the resistance of the membrane potential.
        sharpness (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("ELIF")
        self._initialize(neurons)

    def _dv_dt(self, neurons):
        """
        Single step voltage dynamics of exponential LIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        simple = super(ELIF, self)._dv_dt(neurons)
        v_m = (neurons.v - neurons.theta_rh) / neurons.sharpness
        nonlinear_f = neurons.sharpness * np.exp(v_m)
        return simple + nonlinear_f

    def new_iteration(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self._fire(neurons)

        neurons.v += self._dv_dt(neurons) / neurons.tau


class AELIF(ELIF):
    """
    The neural dynamics of Adaptive Exponential LIF is defined by:

    F(u) = sharpness * exp((v - theta_rh) / sharpness),
    tau*dv/dt = v_rest - v + F(u) + R*I - R*A,
    tau_a*dA/dt = alpha*(v - v_rest) - A + beta*tau_a*spikes,
    if v >= threshold then v = v_reset.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        v_rest (float): voltage at rest.
        v_reset (float): value of voltage reset.
        threshold (float): the voltage threshold.
        R (float): the resistance of the membrane potential.
        sharpness (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
        alpha (float): subthreshold adaptation parameter.
        beta (float): spike-triggered adaptation parameter.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("AELIF")
        self._initialize(neurons)
        neurons.A = neurons.get_neuron_vec(mode="zeros()")

    def _dA_dt(self, neurons):
        """
        Single step adaptation dynamics of AELIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        spike_adaptation = neurons.beta * neurons.tau_a * neurons.spikes
        sub_thresh_adaptation = neurons.alpha * (neurons.v - neurons.v_rest)
        return sub_thresh_adaptation + spike_adaptation - neurons.A

    def new_iteration(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self._fire(neurons)

        dv_dt = self._dv_dt(neurons) - neurons.R * neurons.A
        neurons.v += dv_dt / neurons.tau

        neurons.A = self._dA_dt(neurons) / neurons.tau_a
