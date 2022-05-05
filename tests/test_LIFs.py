from PymoNNto import Network, NeuronGroup, get_squared_dim

from PymoNNtoBehaviors.neurons.LIFs import AELIF, ELIF, LIF


class TestLIFs:
    """
    Tests all variants of LIF neurons defined.
    """

    def test_init(self):
        network = Network()

        params = {
            LIF: {
                "v_rest": -65.0,
                "v_reset": -65.0,
                "threshold": -55.0,
                "R": 1.0,
                "I": 80.0,
                "tau": 10.0,
            },
            ELIF: {
                "v_rest": -65.0,
                "v_reset": -65.0,
                "threshold": -55.0,
                "R": 1.0,
                "I": 80.0,
                "tau": 10.0,
                "sharpness": 5.0,
                "theta_rh": -58.0,
            },
            AELIF: {
                "v_rest": -65.0,
                "v_reset": -65.0,
                "threshold": -55.0,
                "R": 1.0,
                "I": 80.0,
                "tau": 10.0,
                "sharpness": 5.0,
                "theta_rh": -58.0,
                "alpha": 1.0,
                "beta": 2.0,
            },
        }

        for neurons_behavior in params:
            for n in [1, 10, 100, 10000]:
                neurons = NeuronGroup(
                    net=network,
                    tag=f"{neurons_behavior}_population",
                    size=get_squared_dim(n),
                    behaviour={1: neurons_behavior(**params[neurons_behavior])},
                )
                network.initialize()

                assert (neurons.spikes == 0).all()
                assert (neurons.v == params[neurons_behavior]["v_rest"]).all()

                for param in params[neurons_behavior]:
                    assert getattr(neurons, param) == params[neurons_behavior][param]

                # TODO: test the dynamics


if __name__ == "__main__":
    tester = TestLIFs()
    tester.test_init()
