from enum import IntEnum
from typing import Literal

import cv2
import numpy as np


class BayerCFA(IntEnum):
    RGGB = 0b00
    GRBG = 0b01
    GBRG = 0b10
    BGGR = 0b11


def positions(cfa: BayerCFA) -> tuple[int, int, int, int]:
    redx = cfa & 0b01
    redy = (cfa >> 1) & 0b01
    bluex = 1 - redx
    bluey = 1 - redy
    return redx, redy, bluex, bluey


def red2cfa(redx: int, redy: int) -> BayerCFA:
    if not (redx == 0 or redx == 1) or not (redy == 0 or redy == 1):
        raise ValueError("The positions of the red pixels must be 0 or 1.")
    return BayerCFA((redy << 1) | redx)


def pack(
    img: np.ndarray, cfa_type: Literal["RGGB", "GRBG", "GBRG", "BGGR"]
) -> np.ndarray:
    height, width, channels = np.atleast_3d(img).shape
    if channels != 1:
        raise ValueError("A CFA must only have 1 channel.")
    redx, redy, bluex, bluey = positions(BayerCFA[cfa_type])
    out = np.empty_like(img, shape=(height // 2, width // 2, 4))
    out[:, :, 0] = img[redy::2, redx::2]
    out[:, :, 1] = img[redy::2, bluex::2]
    out[:, :, 2] = img[bluey::2, bluex::2]
    out[:, :, 3] = img[bluey::2, redx::2]
    return out


def unpack(
    img: np.ndarray, cfa_type: Literal["RGGB", "GRBG", "GBRG", "BGGR"]
) -> np.ndarray:
    height, width, channels = img.shape
    if channels != 4:
        raise ValueError("A packed CFA must have 4 channels.")
    redx, redy, bluex, bluey = positions(BayerCFA[cfa_type])
    out = np.empty_like(img, shape=(height * 2, width * 2))
    out[redy::2, redx::2] = img[:, :, 0]
    out[redy::2, bluex::2] = img[:, :, 1]
    out[bluey::2, bluex::2] = img[:, :, 2]
    out[bluey::2, redx::2] = img[:, :, 3]
    return out


def black_level_correction(img: np.ndarray, black: int | float) -> np.ndarray:
    return img.astype(np.float32) - black


def lens_shading_correction(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    if img.shape[:2] != gain.shape[:2]:
        raise ValueError(
            "The image and gain matrix must have the same spatial dimensions."
        )

    return np.einsum(
        "hwc,hw->hwc", np.atleast_3d(img.astype(np.float32)), gain.astype(np.float32)
    )


def white_balance(
    img: np.ndarray,
    wb_values: np.ndarray,
    cfa_type: Literal["RGGB", "GRBG", "GBRG", "BGGR"] = None,
) -> np.ndarray:
    if cfa_type is None:
        return white_balance_channels(img.astype(np.float32), wb_values)
    else:
        return white_balance_cfa(img.astype(np.float32), wb_values, cfa_type)


def white_balance_cfa(
    img: np.ndarray,
    wb_values: np.ndarray,
    cfa_type: Literal["RGGB", "GRBG", "GBRG", "BGGR"],
) -> np.ndarray:
    if np.atleast_3d(img).shape[-1] != 1:
        raise ValueError("A CFA must only have 1 channel.")
    if len(wb_values) != 4:
        raise ValueError("4 values are required when white balancing a CFA.")
    redx, redy, bluex, bluey = positions(cfa_type)
    out = np.empty_like(img)
    out[redy::2, redx::2] = img[redy::2, redx::2] * wb_values[0]
    out[redy::2, bluex::2] = img[redy::2, bluex::2] * wb_values[1]
    out[bluey::2, bluex::2] = img[bluey::2, bluex::2] * wb_values[2]
    out[bluey::2, redx::2] = img[bluey::2, redx::2] * wb_values[3]
    return out


def white_balance_channels(img: np.ndarray, wb_values: np.ndarray) -> np.ndarray:
    if wb_values.size != img.shape[-1]:
        raise ValueError(
            f"The number of white balance values ({wb_values.size}) must be equal to the number of channels ({img.shape[-1]})."
        )
    return img * wb_values


def demosaic(img, pattern: Literal["RGGB", "GRBG", "GBRG", "BGGR"]):
    if pattern == "BGGR":
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    elif pattern == "RGGB":
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
    elif pattern == "GBRG":
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    elif pattern == "GRBG":
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
    return img


def camera_color_correction(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,klj->kli", matrix, img.astype(np.float32))


def gamma_correction(img: np.ndarray, gamma: float, white: int | float) -> np.ndarray:
    return np.clip(img.astype(np.float32) / white, 0, 1) ** (1 / gamma)


def tonemap_smoothstep(img: np.ndarray, white: int | float) -> np.ndarray:
    img_norm = np.clip(img.astype(np.float32) / white, 0, 1)
    return 3 * (img_norm**2) - 2 * (img_norm**3)
