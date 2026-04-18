from typing import Literal, override

import albumentations.augmentations.pixel.functional as fpixel
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from einops import pack, repeat

from ..isp.sensor import sensors_dict

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float16"): 1.0,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.float16: 1.0,
    np.float32: 1.0,
    np.float64: 1.0,
    np.int32: 2147483647,
}


class GaussNoise(ImageOnlyTransform):
    def __init__(
        self,
        mean_range: tuple[float, float],
        std_range: tuple[float, float],
        per_channel: bool = True,
        noise_scale_factor: float = 1,
        concat_std_map: bool = False,
        no_noise: bool = False,
        clip: bool = True,
        seed: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.mean_range = mean_range
        self.std_range = std_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor
        self.concat_std_map = concat_std_map
        self.no_noise = no_noise
        self.clip = clip
        self.set_random_seed(seed)

    @override
    def apply(self, img: np.ndarray, mean_value: float, std_value: float, **params):
        height, width, _channels = img.shape
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]

        noise_map = fpixel.generate_noise(
            noise_type="gaussian",
            spatial_mode="per_pixel" if self.per_channel else "shared",
            shape=img.shape,
            params={
                "mean_range": (mean_value, mean_value),
                "std_range": (std_value, std_value),
            },
            max_value=max_value,
            approximation=self.noise_scale_factor,
            random_generator=self.random_generator,
        )

        if not self.no_noise:
            if self.clip:
                noisy = fpixel.add_noise(img, noise_map)
            else:
                noisy = img + noise_map
                noisy = noisy.astype(img.dtype)
        else:
            noisy = img

        if self.concat_std_map:
            std_map = repeat(
                np.array([std_value], dtype=img.dtype),
                "one -> h w one",
                one=1,
                h=height,
                w=width,
            )
            noisy, _ = pack([noisy, std_map], "h w *")
        return noisy

    @override
    def get_params(self):
        sigma = self.py_random.uniform(*self.std_range)
        mean = self.py_random.uniform(*self.mean_range)
        return {"mean_value": mean, "std_value": sigma}


class SensorNoise(ImageOnlyTransform):
    def __init__(
        self,
        sensor: Literal[
            "CRVD", "SamsungS6", "iPhone7", "Pixel", "Nexus6", "LGG4", "IMX586"
        ] = None,
        iso: int | None = None,
        concat_std_map: bool = False,
        no_noise: bool = False,
        clip: bool = True,
        seed: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.sensor = sensors_dict.get(sensor)
        self.iso = iso
        self.concat_std_map = concat_std_map
        self.no_noise = no_noise
        self.clip = clip
        self.set_random_seed(seed)

    @override
    def apply(self, img: np.ndarray, sigma_shot: float, sigma_read: float, **params):
        if not self.no_noise:
            noisy = (
                self.random_generator.poisson(img / sigma_shot, size=img.shape)
                * sigma_shot
            )
            noisy = noisy + self.random_generator.normal(
                0.0, np.sqrt(sigma_read), size=img.shape
            )
            if self.clip:
                noisy = np.clip(noisy, 0.0, 1.0)
        else:
            noisy = img

        if self.concat_std_map:
            std_map = np.sqrt(np.maximum(sigma_shot * noisy + sigma_read, 0.0))
            noisy, _ = pack([noisy, std_map], "h w *")
        return noisy.astype(img.dtype)

    @override
    def get_params(self):
        sensor = self.sensor
        if sensor is None:
            sensor_name = self.py_random.choice(list(sensors_dict.keys()))
            sensor = sensors_dict[sensor_name]

        iso = self.iso
        if iso is None:
            iso = self.py_random.choice(np.arange(sensor.isos[0], sensor.isos[-1] + 1))

        sigma_shot, sigma_read = sensor.noise_model(iso)
        return {"sigma_shot": sigma_shot, "sigma_read": sigma_read}


class DNDNoise(ImageOnlyTransform):
    NOISE_PARAMS = np.array(
        [
            [1.15582540e-02, 1.06784058e-03],
            [3.18601729e-03, 6.66087904e-05],
            [9.44148024e-04, 9.31705956e-06],
            [1.69430208e-03, 1.96081983e-05],
            [1.69430208e-03, 1.96081983e-05],
            [6.69231563e-04, 2.48064038e-06],
            [3.51654607e-04, 3.09506650e-07],
            [3.51654607e-04, 3.09506650e-07],
            [2.66226597e-03, 3.75145954e-05],
            [3.69672139e-03, 8.96344178e-05],
            [1.06288495e-04, 5.65291659e-08],
            [1.31596162e-03, 1.00886736e-05],
            [6.11773651e-03, 2.56685991e-04],
            [1.86896695e-04, 9.96703076e-08],
            [1.31596162e-03, 1.00886736e-05],
            [3.69672139e-03, 8.96344178e-05],
            [7.63732326e-04, 3.53383427e-06],
            [2.29730460e-04, 1.51313674e-07],
            [1.47152058e-03, 1.49382035e-05],
            [4.12791160e-04, 7.46259985e-07],
            [2.87833361e-03, 6.14552159e-05],
            [1.37497189e-04, 5.26281845e-08],
            [2.87833361e-03, 6.14552159e-05],
            [2.29730460e-04, 1.51313674e-07],
            [1.47152058e-03, 1.49382035e-05],
            [2.87833361e-03, 6.14552159e-05],
            [6.28082469e-03, 2.99076866e-04],
            [2.87833361e-03, 6.14552159e-05],
            [2.29730460e-04, 1.51313674e-07],
            [7.42819211e-04, 7.48206523e-07],
            [1.36968732e-03, 2.50255431e-06],
            [3.94269103e-04, 2.80505625e-07],
            [3.54970453e-03, 4.10569107e-05],
            [7.42819211e-04, 7.48206523e-07],
            [3.94269103e-04, 2.80505625e-07],
            [1.36968732e-03, 2.50255431e-06],
            [3.94269103e-04, 2.80505625e-07],
            [2.33877960e-04, 1.52113063e-07],
            [2.61795284e-03, 1.47297540e-05],
            [1.86896695e-04, 9.96703076e-08],
            [3.69672139e-03, 8.96344178e-05],
            [1.31596162e-03, 1.00886736e-05],
            [6.11773651e-03, 2.56685991e-04],
            [9.44148024e-04, 9.31705956e-06],
            [1.69430208e-03, 1.96081983e-05],
            [5.71963750e-04, 5.92547568e-06],
            [1.15582540e-02, 1.06784058e-03],
            [9.44148024e-04, 9.31705956e-06],
        ]
    )

    def __init__(
        self,
        concat_std_map: bool = False,
        no_noise: bool = False,
        clip: bool = True,
        seed: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.concat_std_map = concat_std_map
        self.no_noise = no_noise
        self.clip = clip
        self.set_random_seed(seed)

    @override
    def apply(self, img: np.ndarray, sigma_shot: float, sigma_read: float, **params):
        if not self.no_noise:
            noisy = (
                self.random_generator.poisson(img / sigma_shot, size=img.shape)
                * sigma_shot
            )
            noisy = noisy + self.random_generator.normal(
                0.0, np.sqrt(sigma_read), size=img.shape
            )
            if self.clip:
                noisy = np.clip(noisy, 0.0, 1.0)
        else:
            noisy = img

        if self.concat_std_map:
            std_map = np.sqrt(np.maximum(sigma_shot * noisy + sigma_read, 0.0))
            noisy, _ = pack([noisy, std_map], "h w *")
        return noisy.astype(img.dtype)

    @override
    def get_params(self):
        id = self.py_random.randrange(len(self.NOISE_PARAMS))
        sigma_shot, sigma_read = self.NOISE_PARAMS[id]
        return {"sigma_shot": sigma_shot, "sigma_read": sigma_read}


class DNDCameraNoise(ImageOnlyTransform):
    CAMERAS = {
        "CanonEOS5D4": {
            "Kmin": 0.0002,
            "Kmax": 0.02,
            "slope": 0.6771267783987617,
            "bias": 1.5121876510805845,
            "sigma": 0.24641096601611254,
        },
        "CanonEOS700D": {
            "Kmin": 0.0002,
            "Kmax": 0.02,
            "slope": 0.6524587630931787,
            "bias": 2.079863926921898,
            "sigma": 0.25781594662342805,
        },
        "CanonEOS70D": {
            "Kmin": 0.0002,
            "Kmax": 0.02,
            "slope": 0.5681183926808191,
            "bias": 1.9982663724918939,
            "sigma": 0.24870223821017215,
        },
        "SonyA7S2": {
            "Kmin": 0.0002,
            "Kmax": 0.02,
            "slope": 0.5407496082896145,
            "bias": 1.218237123334061,
            "sigma": 0.26751211630129734,
        },
        "NikonD850": {
            "Kmin": 0.0002,
            "Kmax": 0.02,
            "slope": 0.5634612649498357,
            "bias": 0.799410107756515,
            "sigma": 0.21408231874600314,
        },
    }

    def __init__(
        self,
        concat_std_map: bool = False,
        no_noise: bool = False,
        clip: bool = True,
        seed: int | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.concat_std_map = concat_std_map
        self.no_noise = no_noise
        self.clip = clip
        self.set_random_seed(seed)

    @override
    def apply(self, img: np.ndarray, sigma_shot: float, sigma_read: float, **params):
        if not self.no_noise:
            noisy = (
                self.random_generator.poisson(img / sigma_shot, size=img.shape)
                * sigma_shot
            )
            noisy = noisy + self.random_generator.normal(
                0.0, np.sqrt(sigma_read), size=img.shape
            )
            if self.clip:
                noisy = np.clip(noisy, 0.0, 1.0)
        else:
            noisy = img

        if self.concat_std_map:
            std_map = np.sqrt(np.maximum(sigma_shot * noisy + sigma_read, 0.0))
            noisy, _ = pack([noisy, std_map], "h w *")
        return noisy.astype(img.dtype)

    @override
    def get_params(self):
        camera = self.py_random.choice(list(self.CAMERAS.keys()))
        camera_data = self.CAMERAS[camera]
        log_a = self.py_random.uniform(
            np.log(camera_data["Kmin"]), np.log(camera_data["Kmax"])
        )
        a = np.exp(log_a)

        log_b = (
            self.py_random.gauss() * camera_data["sigma"]
            + (camera_data["slope"] + 2) * log_a
            + camera_data["bias"]
        )
        b = np.exp(log_b)
        return {"sigma_shot": a, "sigma_read": b}
