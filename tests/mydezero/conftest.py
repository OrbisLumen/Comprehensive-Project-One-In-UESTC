import numpy as np
import pytest


@pytest.fixture
def regression_model():
    """Return a fresh one-layer model with zero weights and an unused parameter."""
    from mydezero import Model, Parameter
    from mydezero.layers import Linear

    class Regression(Model):
        def __init__(self):
            super().__init__()
            self.linear = Linear(1, dtype=np.float64, in_size=1)
            self.linear.W.data[:] = 0
            self.unused = Parameter(np.array(7.0))

        def forward(self, x):
            return self.linear(x)

    return Regression()
