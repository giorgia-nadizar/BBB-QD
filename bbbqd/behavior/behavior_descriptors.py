import numpy as np


def _compute_spectrum(signal: np.ndarray) -> np.ndarray:
    fft = np.fft.fft(signal)
    abs_fft = np.abs(fft)
    half_fft = abs_fft[:int(len(signal) / 2) + 1]
    return half_fft


def _compute_spectra(signal: np.ndarray) -> np.ndarray:
    return np.asarray([_compute_spectrum(signal[:, i]) for i in range(signal.shape[1])])


def _signal_peak(signal: np.ndarray, ignore_continuous_component: bool = False, cut_off: float = 0.4) -> float:
    if ignore_continuous_component:
        signal = signal[1:len(signal)]
    if len(signal) == 0:
        return 0
    peak_freq = np.argmax(signal) / len(signal)
    relative_peak_freq = peak_freq / cut_off
    return min(relative_peak_freq, 1.)


def _signal_median(signal: np.ndarray, ignore_continuous_component: bool = False, cut_off: float = 0.4) -> float:
    if ignore_continuous_component:
        signal = signal[1:len(signal)]
    if len(signal) == 0:
        return 0
    total_sum = np.sum(signal)
    first_sum = 0
    median_idx = len(signal - 1)
    for idx, value in enumerate(signal):
        first_sum += value
        if first_sum > (total_sum / 2):
            median_idx = idx
            break
    median_freq = median_idx / len(signal)
    relative_median_freq = median_freq / cut_off
    return min(relative_median_freq, 1.)
