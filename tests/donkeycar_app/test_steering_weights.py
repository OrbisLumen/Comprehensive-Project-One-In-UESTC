import numpy as np
import pytest

from donkeycar_app.dataset import steering_loss_weights


def test_large_commands_preserve_spike_downweighting_and_original_labels():
    rows = [dict(raw=value, target=0.0, weight=weight, spike=spike)
            for value, weight, spike in [(0.29, 1, False), (0.3, 1, False),
                                         (-0.6, 1, False), (1.0, 0.1, True)]]
    before = [row.copy() for row in rows]
    np.testing.assert_allclose(steering_loss_weights(rows, 4).ravel(), [1, 4, 4, 0.1])
    np.testing.assert_allclose(steering_loss_weights(rows, 1).ravel(), [1, 1, 1, 0.1])
    assert rows == before


@pytest.mark.parametrize("multiplier,threshold", [(0, 0.3), (float('nan'), 0.3),
                                                   (2, 0), (2, float('inf'))])
def test_invalid_weight_settings(multiplier, threshold):
    with pytest.raises(ValueError):
        steering_loss_weights([], multiplier, threshold)
