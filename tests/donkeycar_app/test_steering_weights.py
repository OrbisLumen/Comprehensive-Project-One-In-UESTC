import numpy as np
import pytest

from donkeycar_app.dataset import STEERING_RECIPES, steering_loss_weights


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


@pytest.mark.parametrize("family,label,spike_weight", [
    ("raw", "raw", 1.0),
    ("downweighted", "raw", 0.1),
    ("smoothed", "target", 1.0),
])
@pytest.mark.parametrize("multiplier", [1, 2, 4])
def test_training_recipe_labels_and_weights(family, label, spike_weight, multiplier):
    name = "raw" if (family, multiplier) == ("raw", 1) else f"{family}_x{multiplier}"
    spec = STEERING_RECIPES[name]
    rows = [dict(raw=0.29, target=0.29, weight=1, spike=False),
            dict(raw=-1, target=0.3, weight=0.1, spike=True),
            dict(raw=0.3, target=0.3, weight=1, spike=False)]
    weights = steering_loss_weights(rows, spec.sharp_multiplier, downweight_spikes=spec.downweight_spikes)
    assert spec.label == label
    np.testing.assert_allclose(weights[:, 0], [1, spike_weight, multiplier])
