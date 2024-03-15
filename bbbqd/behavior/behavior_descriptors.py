import numpy as np


def _compute_spectrum(signal: np.ndarray) -> np.ndarray:
    fft = np.fft.fft(signal)
    abs_fft = np.abs(fft)
    half_fft = abs_fft[:int(len(signal) / 2) + 1]
    return half_fft
