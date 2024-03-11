import math

import numpy as np
from typing import Tuple


def compute_spectrum(signal: np.ndarray, delta_t: float) -> Tuple[np.ndarray, np.ndarray]:
    padded_size = 2 ** (math.ceil(math.log(len(signal) / math.log(2))))
    fft = np.fft.fft(signal, n=padded_size)
    abs_fft = np.abs(fft)
    half_fft = abs_fft[:int(padded_size / 2) + 1]
    frequencies = [1. / delta_t / 2. * i / len(half_fft) for i in range(len(half_fft))]
    return half_fft, np.asarray(frequencies)
