# src/filter_utils.py
import numpy as np
from scipy import signal

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Terapkan band-pass Butterworth filter secara zero-phase.
    Params:
      data    : 1D array sinyal
      lowcut  : frekuensi cutoff bawah (Hz)
      highcut : frekuensi cutoff atas (Hz)
      fs      : sampling rate (Hz)
      order   : orde filter
    Return:
      filtered_data : 1D array hasil filtering
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)
