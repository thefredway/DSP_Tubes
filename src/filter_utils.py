import numpy as np
from scipy import signal

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Menerapkan filter band-pass Butterworth pada sinyal.
    
    Parameter:
    - data: array 1D dari sinyal masukan
    - lowcut: frekuensi batas bawah (Hz)
    - highcut: frekuensi batas atas (Hz)
    - fs: frekuensi sampling (Hz)
    - order: orde filter (default = 5)

    Return:
    - filtered_data: sinyal hasil filtering
    """
    nyq = 0.5 * fs # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band') # Desain filter
    return signal.filtfilt(b, a, data) # Terapkan zero-phase filter
