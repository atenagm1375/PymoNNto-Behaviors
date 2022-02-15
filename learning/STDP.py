"""
Implementation of Spike-Time Dependent Plasticity (STDP) based on local variables.
"""

from PymoNNto import Behaviour
import numpy as np


class STDP(Behaviour):
    """
    Let x and y be the spike trace variables of pre- and post-synaptic neurons.
    These trace variables are modified through time with:

    dx/dt = -x/tau_plus + pre.spikes,
    dy/dt = -y/tau_minus + post.spikes,

    where tau_plus and tau_minus define the time window of STDP. Then, the synaptic
    weights can are updated with the given equation:

    dw/dt = a_plus * x * post.spikes - a_minus * y * pre.spikes,

    where a_plus and a_minus define the intensity of weight change.

    We assume that synapses have dt, w_min, and w_max and neurons have spikes.

    Args:
        tau_plus (float): pre-post time window.
        tau_minus (float): post-pre time window.
        a_plus (float): pre-post intensity.
        a_minus (float): post-pre intensity.
    """
    def set_variables(self, synapses):
        """
        Set STDP variables and pre- and post-synaptic neuron traces.

        Args:
            synapses (SynapseGroup): the synapse to which STDP is applied.
        """
        self.add_tag('STDP')
        self.set_init_attrs_as_variables(synapses)
        synapses.src.trace = synapses.src.get_neuron_vec()
        synapses.dst.trace = synapses.dst.get_neuron_vec()
    
    def new_iteration(self, synapses):
        """
        Single step of STDP.

        Args:
            synapses (SynapseGroup): the synapse to which STDP is applied.
        """
        dx = -synapses.src.trace / synapses.tau_plus + synapses.src.spikes
        dy = -synapses.dst.trace / synapses.tau_minus + synapses.dst.spikes
        synapses.src.trace += dx * synapses.dt
        synapses.dst.trace += dy * synapses.dt

        dw_minus = -synapses.a_minus * synapses.dst.trace * synapses.src.spikes
        dw_plus = synapses.a_plus * synapses.src.trace * synapses.dst.spikes

        synapses.W = np.clip(
            synapses.W + (dw_plus + dw_minus) * synapses.dt,
            synapses.w_min, synapses.w_max
            )
