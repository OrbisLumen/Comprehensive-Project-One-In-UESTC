"""Course AutoDriveNet for single-frame steering on Generated Road."""

import cv2
import numpy as np
import torch
from torch import nn


def hsv_frame(rgb):
    """Convert a 120 x 160 RGB uint8 simulator frame to OpenCV HSV."""
    if rgb.shape != (120, 160, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected a 120 x 160 RGB uint8 frame")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


class AutoDriveNet(nn.Module):
    """Map normalized NCHW HSV images to one steering command per image."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.Dropout(0.5),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 13, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(), nn.Linear(50, 10), nn.Linear(10, 1),
        )

    def forward(self, images):
        return self.regressor(self.features(images)).flatten()

    @torch.inference_mode()
    def steer(self, rgb):
        """Predict a bounded steering value in evaluation mode."""
        pixels = torch.from_numpy(hsv_frame(rgb)).to(next(self.parameters()).device)
        batch = pixels.permute(2, 0, 1).unsqueeze(0).float().div_(255)
        return self(batch).clamp(-1, 1).item()
