from dataclasses import dataclass

import numpy as np


@dataclass
class SampledSensor:
    isos: np.ndarray
    sigmas_shot: np.ndarray
    sigmas_read: np.ndarray

    def noise_model(self, iso: int):
        sigma_shot = np.interp(iso, self.isos, self.sigmas_shot)
        sigma_read = np.interp(iso, self.isos, np.sqrt(self.sigmas_read)) ** 2
        return sigma_shot, sigma_read


@dataclass
class CurveFittedSensor:
    isos: tuple[int, int]
    sigma_shot_fit: np.polynomial.Polynomial
    sigma_read_fit: np.polynomial.Polynomial

    def noise_model(self, iso: int):
        sigma_shot = self.sigma_shot_fit(iso)
        sigma_read = self.sigma_read_fit(iso)
        return sigma_shot, sigma_read


IMX586Sensor = CurveFittedSensor(
    isos=(100, 6400),
    sigma_shot_fit=np.polynomial.Polynomial([0.00868861, 0.0005995267]) / 255,
    sigma_read_fit=np.polynomial.Polynomial([0.11492713, 6.514934e-4, 7.11772e-7])
    / 255
    / 255,
)


CRVDSensor = SampledSensor(
    isos=np.array([100, 1600, 3200, 6400, 12800, 25600]),
    sigmas_shot=np.array([0.22, 3.513262, 6.955588, 13.486051, 26.585953, 52.032536])
    / 4095,
    sigmas_read=np.array(
        [0.0466, 11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
    )
    / 4095
    / 4095,
)

SamsungS6Sensor = SampledSensor(
    isos=np.array([100, 200, 400, 800, 1600, 3200]),
    sigmas_shot=np.array(
        [
            0.0005928246,
            0.0011573000,
            0.0020248293,
            0.0042518379,
            0.0087945818,
            0.0175841665,
        ]
    ),
    sigmas_read=np.array(
        [0.0, 1.584088e-06, 2.204615e-05, 3.955212e-05, 1.307663e-04, 4.811261e-04]
    ),
)

IPhone7Sensor = SampledSensor(
    isos=np.array([100, 200, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000]),
    sigmas_shot=np.array(
        [
            0.0007217942,
            0.0012467039,
            0.0012800903,
            0.0013023417,
            0.0012330618,
            0.0012494413,
            0.0012196905,
            0.0012070881,
            0.0013796850,
            0.0006992866,
            0.0012842782,
        ]
    ),
    sigmas_read=np.array(
        [
            5.765596e-06,
            5.439449e-06,
            4.655187e-06,
            4.694002e-06,
            5.931989e-06,
            5.118882e-06,
            5.796433e-06,
            5.735137e-06,
            3.947180e-06,
            1.053361e-05,
            4.526013e-06,
        ]
    ),
)

PixelSensor = SampledSensor(
    isos=np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]),
    sigmas_shot=np.array(
        [
            0.0001183842,
            0.0002653817,
            0.0004169289,
            0.0008872063,
            0.0016859508,
            0.0032797598,
            0.0070585537,
            0.0145326996,
            0.0228839011,
        ]
    ),
    sigmas_read=np.array(
        [
            1.373917e-06,
            2.619070e-06,
            3.109200e-06,
            5.297723e-07,
            1.417511e-05,
            4.695371e-05,
            1.617292e-04,
            4.978458e-04,
            1.317243e-03,
        ]
    ),
)

Nexus6Sensor = SampledSensor(
    isos=np.array([100, 400, 800, 1600, 3200]),
    sigmas_shot=np.array(
        [0.0004526757, 0.0015939748, 0.0032301480, 0.0067342243, 0.0137550188]
    ),
    sigmas_read=np.array(
        [3.217001e-06, 1.326825e-05, 3.054194e-05, 1.243178e-04, 5.384936e-04]
    ),
)

LGG4Sensor = SampledSensor(
    isos=np.array([100, 200, 400, 800]),
    sigmas_shot=np.array([0.0003938314, 0.0007271523, 0.0014359446, 0.0028555957]),
    sigmas_read=np.array([3.041726e-06, 4.027626e-06, 9.612015e-06, 3.138330e-05]),
)

sensors_dict = {
    "CRVD": CRVDSensor,
    "SamsungS6": SamsungS6Sensor,
    "iPhone7": IPhone7Sensor,
    "Pixel": PixelSensor,
    "Nexus6": Nexus6Sensor,
    "LGG4": LGG4Sensor,
    "IMX586": IMX586Sensor,
}
