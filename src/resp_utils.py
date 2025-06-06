import cv2
import numpy as np
import mediapipe as mp

def create_pose_landmarker(model_path: str, use_gpu: bool=False):
    """
    Memuat model `pose_landmarker` untuk deteksi pose tubuh menggunakan MediaPipe.
    
    Parameter:
    - model_path: path ke file .task
    - use_gpu: jika True maka menggunakan GPU, default CPU

    Return:
    - objek PoseLandmarker yang sudah diinisialisasi
    """
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    with open(model_path, "rb") as f:
        model_content = f.read()

    base_options = BaseOptions(
        model_asset_buffer=model_content,
        delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
    )

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print(f"[DEBUG] Loaded model from buffer: {model_path}")
    return PoseLandmarker.create_from_options(options)


class RespTracker:
    """
    Pelacak sinyal respirasi berdasarkan Optical Flow pada ROI bahu.
    """
    def __init__(self, landmarker, x_size=100, y_size=100, shift_x=0, shift_y=0):
        self.landmarker = landmarker
        self.x_size = x_size
        self.y_size = y_size
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.features = None
        self.old_gray = None
        self.shoulder_pts = None  # (x1, y1), (x2, y2)
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.roi = None  # (left, top, right, bottom)

    def initialize(self, frame: np.ndarray, timestamp_ms: int):
        """
        Deteksi awal bahu dan pilih titik fitur untuk Optical Flow.
        Params:
          frame        : frame awal
          timestamp_ms : waktu frame dalam milidetik (dibutuhkan oleh pose model)
        """
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = self.landmarker.detect_for_video(mp_img, timestamp_ms=timestamp_ms)
        if not res.pose_landmarks:
            raise RuntimeError("Pose tidak terdeteksi.")

        lm = res.pose_landmarks[0]
        ls, rs = lm[11], lm[12]
        cx = int((ls.x + rs.x) * w / 2) + self.shift_x
        cy = int((ls.y + rs.y) * h / 2) + self.shift_y
        l = max(0, cx - self.x_size)
        r = min(w, cx + self.x_size)
        t = max(0, cy - self.y_size)
        b = min(h, cy + self.y_size)
        self.roi = (l, t, r, b)
        self.shoulder_pts = [(int(ls.x * w), int(ls.y * h)), (int(rs.x * w), int(rs.y * h))]


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = gray.copy()
        chest = gray[t:b, l:r]
        pts = cv2.goodFeaturesToTrack(chest, maxCorners=1000, qualityLevel=0.01, minDistance=3, blockSize=7)
        if pts is None:
            raise RuntimeError("Gagal menemukan feature untuk tracking.")
        pts[:, :, 0] += l
        pts[:, :, 1] += t
        self.features = np.float32(pts)

    def update(self, frame: np.ndarray) -> float:
        """
        Melacak Optical Flow dan mengembalikan posisi vertikal rata-rata.

        Parameter:
        - frame: frame gambar

        Return:
        - nilai rata-rata posisi y dari fitur pelacakan
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.features, None, **self.lk_params)
        good_new = new_pts[status == 1].reshape(-1, 2)
        self.features = good_new.reshape(-1, 1, 2)
        self.old_gray = gray
        return float(np.mean(good_new[:, 1]))
