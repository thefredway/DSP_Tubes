# src/rppg_utils.py
import numpy as np
from filter_utils import bandpass_filter

def cpu_POS(X: np.ndarray, fps: float) -> np.ndarray:
    """
    POS algorithm (Wang et al. 2016).
    Input X shape = (e, 3, f), output H shape = (e, f).
    """
    eps = 1e-9
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P]*e, axis=0)  # shape (e,2,3)
    H = np.zeros((e, f))
    for n in range(w, f):
        m = n - w + 1
        Cn = X[:, :, m:n+1]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        Cn = Cn * M[:, :, None]
        S = np.tensordot(Q, Cn, axes=([2],[1]))  # (e,2,w)
        S = np.swapaxes(S[0], 0, 1)               # (w,2) -> (2,w)
        S1, S2 = S[:, :], S[:, :]
        alpha = np.std(S1, axis=1)/(eps + np.std(S2, axis=1))
        Hn = S1 + alpha[:,None]*S2
        Hnm = Hn - np.mean(Hn, axis=1)[:,None]
        H[:, m:n+1] += Hnm
    return H

def extract_rppg(rgb_buffer: np.ndarray, fps: float,
                 lowcut: float = 0.8, highcut: float = 2.5,
                 filter_order: int = 5) -> np.ndarray:
    """
    Hitung rPPG dari buffer RGB.
    Params:
      rgb_buffer   : array shape (3, f) dengan urutan [R;G;B]
      fps          : frame rate
      lowcut/highcut : batas filter (Hz)
    Return:
      rppg_filtered: sinyal rPPG ter-filter shape (f,)
    """
    # tambahkan dim estimator=1
    sig = rgb_buffer[np.newaxis, ...]         # (1,3,f)
    raw = cpu_POS(sig, fps=fps).flatten()     # (f,)
    return bandpass_filter(raw, lowcut, highcut, fs=fps, order=filter_order)
