import numpy as np
from filter_utils import bandpass_filter

def cpu_POS(X: np.ndarray, fps: float) -> np.ndarray:
    """
    Menghitung sinyal rPPG dengan metode POS (Plane-Orthogonal-to-Skin).
    
    Parameter:
    - X: array (e, 3, f) berisi sinyal RGB dari frame
    - fps: frame per second (sampling rate)

    Return:
    - H: array (e, f) hasil estimasi sinyal POS
    """
    eps = 1e-9
    e, c, f = X.shape
    w = int(1.6 * fps)

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P]*e, axis=0)  # shape (e,2,3)
    H = np.zeros((e, f))

    for n in range(w, f):
        m = n - w + 1
        Cn = X[:, :, m:n+1]                      # (e,3,w)
        M = 1.0 / (np.mean(Cn, axis=2) + eps)    # (e,3)
        Cn = Cn * M[:, :, None]                  # Normalized (e,3,w)
        S = np.tensordot(Q, Cn, axes=([2],[1]))  # (e,2,w)

        Hnm = np.zeros((e, w))
        for i in range(e):
            S1, S2 = S[i, :, :]  # (2,w)
            alpha = np.std(S1) / (np.std(S2) + eps)
            Hn = S1 + alpha * S2
            Hnm[i] = Hn - np.mean(Hn)

        H[:, m:n+1] += Hnm
    return H


def extract_rppg(rgb_buffer: np.ndarray, fps: float,
                 lowcut: float = 0.8, highcut: float = 2.5,
                 filter_order: int = 5) -> np.ndarray:
    """
    Ekstraksi sinyal rPPG dari buffer RGB menggunakan metode POS dan filter bandpass.

    Parameter:
    - rgb_buffer: array (3, f), sinyal RGB
    - fps: frame per second
    - lowcut, highcut: batas frekuensi filter bandpass
    - filter_order: orde filter

    Return:
    - rppg_filtered: sinyal rPPG yang telah difilter
    """
    # tambahkan dim estimator=1
    sig = rgb_buffer[np.newaxis, ...]         # (1,3,f)
    raw = cpu_POS(sig, fps=fps).flatten()     # (f,)
    return bandpass_filter(raw, lowcut, highcut, fs=fps, order=filter_order)
