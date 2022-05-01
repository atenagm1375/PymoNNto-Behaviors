"""
Implementation of Spike-Time Dependent Plasticity (STDP) based on local variables.
"""

import numpy as np
from PymoNNto import Behaviour


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

    def conv2d_update(self, synapses):
        # TODO check if the method works
        src_spikes = synapses.src.spikes
        src_trace = synapses.src.trace
        dst_spikes = synapses.dst.spikes
        dst_trace = synapses.dst.trace

        p0, p1 = synapses.pad
        h_in = int(np.sqrt(synapses.src.size)) + 2 * p0
        w_in = int(np.sqrt(synapses.src.size)) + 2 * p1
        c_in = synapses.src.size // (h_in * w_in)

        pre_spikes = np.zeros((1, h_in, w_in, c_in))
        pre_spikes[:, p0 : h_in - p0, p1 : w_in - p1, :] = src_spikes

        pre_trace = np.zeros((1, h_in, w_in, c_in))
        pre_trace[:, p0 : h_in - p0, p1 : w_in - p1, :] = src_trace

        src_spikes = np.lib.stride_tricks.sliding_window_view(
            pre_spikes, synapses.kernel_size
        )

        src_trace = np.lib.stride_tricks.sliding_window_view(
            pre_trace, synapses.kernel_size
        )

        dst_spikes = dst_spikes.reshape((1, synapses.n_filters, -1))
        dst_trace = dst_trace.reshape((1, synapses.n_filters, -1))

        dw_minus = -synapses.a_minus * dst_trace * src_spikes
        dw_plus = synapses.a_plus * src_trace * dst_spikes

        return dw_minus, dw_plus

    def set_variables(self, synapses):
        """
        Set STDP variables and pre- and post-synaptic neuron traces.

        Args:
            synapses (SynapseGroup): the synapse to which STDP is applied.
        """
        self.add_tag("STDP")
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

        if "conv2d" in synapses.tags:
            dw_minus, dw_plus = self.conv2d_update(synapses)
        else:
            dw_minus = -synapses.a_minus * synapses.dst.trace * synapses.src.spikes
            dw_plus = synapses.a_plus * synapses.src.trace * synapses.dst.spikes

        synapses.W = np.clip(
            synapses.W + (dw_plus + dw_minus) * synapses.dt,
            synapses.w_min,
            synapses.w_max,
        )
