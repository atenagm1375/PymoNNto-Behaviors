import numpy as np
from PymoNNto import Behaviour


class Conv2D(Behaviour):
    """
    TODO add stride support
    TODO add functionalities to work for n*m populations (n!=m)

    Args:
        Behaviour (_type_): _description_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def __conv2d(src, kernels, padding):
        # src.shape = (1, H, W, C)
        # kernels.shape = (h, w, C, n_filters)
        h = src.shape[1] + padding[0] * 2
        w = src.shape[2] + padding[1] * 2

        img = np.zeros((src.shape[0], h, w, src.shape[-1]))
        img[:, padding[0] : h - padding[0], padding[1] : w - padding[1], :] = src

        h_out = int((src.shape[1] - kernels.shape[0] + 2 * padding[0]) + 1)
        w_out = int((src.shape[2] - kernels.shape[1] + 2 * padding[1]) + 1)

        inp = np.lib.stride_tricks.as_strided(
            img,
            (
                img.shape[0],
                h_out,
                w_out,
                kernels.shape[0],
                kernels.shape[1],
                img.shape[3],
            ),
            img.strides[:3] + img.strides[1:],
        )
        return np.tensordot(inp, kernels, axes=3)

    def set_variables(self, synapses):
        self.add_tag("conv2d")
        self.set_init_attrs_as_variables(synapses)

        if synapses.pad:
            synapses.pad = (synapses.kernel_size[0] // 2, synapses.kernel_size[1] // 2)
        else:
            synapses.pad = (0, 0)

        inp = synapses.src.s.reshape(
            (int(np.sqrt(synapses.src.size)), int(np.sqrt(synapses.src.size)))
        )
        self.is1d = synapses.src.s.shape[0] == inp.size

        synapses.W = np.random.uniform(
            synapses.w_min,
            synapses.w_max,
            (
                synapses.kernel_size[0],
                synapses.kernel_size[1],
                synapses.src.s.shape[0] // inp.size,
                synapses.n_filters,
            ),
        )

    def new_iteration(self, synapses):
        inp = synapses.src.s.reshape(
            (int(np.sqrt(synapses.src.size)), int(np.sqrt(synapses.src.size)))
        )
        if self.is1d:
            inp = np.expand_dims(inp, axis=2)

        # trunk-ignore(flake8/E741)
        synapses.dst.I += Conv2D.__conv2d(
            np.expand_dims(inp, axis=0), synapses.W, synapses.pad
        ).flatten()
