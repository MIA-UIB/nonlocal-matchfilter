from typing import Literal

import torch.nn.functional as F


class InputPadder:
    def __init__(
        self,
        dims,
        to_multiple: int,
        position: Literal["symmetric", "downright"] = "symmetric",
        mode: Literal["constant", "reflect", "replicate", "circular"] = "replicate",
        value: float | None = None,
    ):
        self.height, self.width = dims[-2:]
        self.mode = mode
        self.value = value
        pad_h = (
            ((self.height // to_multiple) + 1) * to_multiple - self.height
        ) % to_multiple
        pad_w = (
            ((self.width // to_multiple) + 1) * to_multiple - self.width
        ) % to_multiple
        if position == "symmetric":
            self._pad = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        else:
            self._pad = [0, pad_w, 0, pad_h]

    def pad(self, x):
        return F.pad(x, self._pad, mode=self.mode, value=self.value)

    def unpad(self, x):
        height, width = x.shape[-2:]
        crop_h = [self._pad[2], height - self._pad[3]]
        crop_w = [self._pad[0], width - self._pad[1]]
        return x[..., crop_h[0] : crop_h[1], crop_w[0] : crop_w[1]]
