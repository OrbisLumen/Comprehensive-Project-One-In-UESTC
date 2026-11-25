"""Single-frame steering regression with frozen spatial ResNet50 features."""

import numpy as np
from PIL import Image

from mydezero import cuda, no_grad, test_mode
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.models import ResNet50, VGG16


class SpatialSteeringResNet(ResNet50):
    """Predict normalized steering from mean-centered BGR frames (N, 3, 120, 160)."""

    def __init__(self, pretrained=False):
        super().__init__(pretrained=pretrained)
        self.fc6 = Linear(1, in_size=512 * 3 * 5 + 2048)

    def frozen_features(self, images):
        with test_mode(), no_grad():
            x = F.pooling(F.relu(self.bn1(self.conv1(images))), 3, stride=2)
            mid = self.res3(self.res2(x))
            top = self.res5(self.res4(mid))
            local = mid.data.reshape(len(images), 512, 3, 5, 5, 4).mean(axis=(3, 5))
            global_features = top.data.mean(axis=(2, 3))
            xp = cuda.get_array_module(global_features)
            return xp.concatenate((local.reshape(len(images), -1), global_features), axis=1)

    def forward(self, images):
        return F.clip(self.fc6(self.frozen_features(images)), -1, 1)

    def steer(self, rgb_frame):
        """Predict one steering command from a 120 by 160 RGB camera frame."""
        frame = np.asarray(rgb_frame)
        if frame.shape != (120, 160, 3) or frame.dtype != np.uint8:
            raise ValueError("Expected a uint8 RGB camera frame of shape (120, 160, 3)")
        pixels = VGG16.preprocess(Image.fromarray(frame), size=None)
        xp = cuda.get_array_module(self.fc6.W.data)
        with test_mode(), no_grad():
            return float(self(xp.asarray(np.ascontiguousarray(pixels[None]))).data.item())

