import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from rppg_utils import extract_rppg
from resp_utils import create_pose_landmarker, RespTracker
from filter_utils import bandpass_filter

# --- Parameter ---
FPS        = 30.0
WIN_POS    = int(1.6 * FPS)      # panjang window untuk POS
LOW_RPPG   = 0.8
HIGH_RPPG  = 2.5
LOW_RESP   = 0.1
HIGH_RESP  = 0.7
# -----------------

def main():
    # 1) Buka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka webcam.")

    # 2) Inisialisasi FaceDetector (Tasks API)
    BaseOptions          = mp_tasks.BaseOptions
    FaceDetectorOptions  = vision.FaceDetectorOptions
    VisionRunningMode    = vision.RunningMode
    fd_opts = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/blaze_face_short_range.tflite"),
        running_mode=VisionRunningMode.IMAGE
    )
    face_detector = vision.FaceDetector.create_from_options(fd_opts)

    # 3) Inisialisasi RespTracker
    pose_landmarker = create_pose_landmarker("models/pose_landmarker.task")
    resp_tracker    = RespTracker(
        pose_landmarker,
        x_size=150, y_size=120,
        shift_x=0, shift_y=40
    )

    # Buffers untuk sinyal
    rgb_buffer  = []  # [R, G, B]
    resp_buffer = []  # posisi y bahu

    # Set up interactive plot
    plt.ion()
    fig, (ax_rppg, ax_resp) = plt.subplots(2, 1, figsize=(6, 6))
    ax_rppg.set_title("rPPG (filtered)")
    ax_resp.set_title("Respirasi")
    ax_rppg.set_xlabel("Frame")
    ax_resp.set_xlabel("Frame")
    ax_rppg.grid(True)
    ax_resp.grid(True)

    frame_idx   = 0
    initialized = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 4) Deteksi wajah & ambil ROI
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res     = face_detector.detect(mp_img)

            if res.detections:
                det  = res.detections[0]
                bbox = det.bounding_box
                # origin_x/origin_y dan width/height sudah dalam piksel
                x, y = int(bbox.origin_x), int(bbox.origin_y)
                W, H = int(bbox.width),    int(bbox.height)

                # ambil kotak kecil di tengah dahi
                cx, cy = x + W//2, y + H//2
                R      = min(W, H) // 3
                l, r = cx - R, cx + R
                t, b = cy - R, cy + R

                # clip agar dalam frame
                h, w = frame.shape[:2]
                l, r = max(0, l), min(w, r)
                t, b = max(0, t), min(h, b)

                roi = frame[t:b, l:r]
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                # 5) Ekstraksi rPPG: rata-rata RGB
                mean_bgr = cv2.mean(roi)[:3]
                rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

                # 6) Ekstraksi respirasi: inisialisasi & update tracker
                if not initialized:
                    resp_tracker.initialize(frame)
                    initialized = True
                resp_y = resp_tracker.update(frame)
                resp_buffer.append(resp_y)

            # Tampilkan frame
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 7) Update plot setiap 10 frame, jika buffer sudah cukup
            frame_idx += 1
            if frame_idx % 10 == 0 and len(rgb_buffer) >= WIN_POS:
                # rPPG
                rgb_arr   = np.array(rgb_buffer).T    # shape (3, N)
                rppg_sig  = extract_rppg(rgb_arr, fps=FPS,
                                         lowcut=LOW_RPPG, highcut=HIGH_RPPG)
                # Respirasi
                resp_sig  = bandpass_filter(
                    np.array(resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS
                )

                # Refresh plot
                ax_rppg.cla()
                ax_resp.cla()

                ax_rppg.plot(rppg_sig)
                ax_rppg.set_title("rPPG (filtered)")
                ax_rppg.set_xlabel("Frame")
                ax_rppg.grid(True)

                ax_resp.plot(resp_sig)
                ax_resp.set_title("Respirasi")
                ax_resp.set_xlabel("Frame")
                ax_resp.grid(True)

                fig.canvas.draw()
                plt.pause(0.001)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        pose_landmarker.close()

if __name__ == "__main__":
    main()
