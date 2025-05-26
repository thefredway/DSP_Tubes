import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import os

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from rppg_utils import extract_rppg
from resp_utils import create_pose_landmarker, RespTracker
from filter_utils import bandpass_filter

# --- Parameter ---
FPS        = 30.0
WIN_POS    = int(1.6 * FPS)
LOW_RPPG   = 0.8
HIGH_RPPG  = 2.5
LOW_RESP   = 0.1
HIGH_RESP  = 0.7
# -----------------

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "blaze_face_short_range.tflite"))
pose_path  = os.path.abspath(os.path.join(script_dir, "..", "models", "pose_landmarker.task"))

# Print paths for debugging
print(f"[DEBUG] Script directory: {script_dir}")
print(f"[DEBUG] Face model path: {model_path}")
print(f"[DEBUG] Pose model path: {pose_path}")

def show_countdown_overlay(cap, duration=5):
    hints = [
        "Pastikan pencahayaan cukup",
        "Pastikan wajah terlihat jelas",
        "Pastikan kamera menangkap bahu"
    ]
    for sec in range(duration, 0, -1):
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Mulai dalam {sec}", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (30, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1000)

def main():
    print("[DEBUG] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka webcam.")

    ret_test, frame_test = cap.read()
    print(f"[DEBUG] Initial capture ret={ret_test}, frame_test shape={None if not ret_test else frame_test.shape}")
    if ret_test:
        cv2.imshow("Test Frame", frame_test)
        cv2.waitKey(1000)
        cv2.destroyWindow("Test Frame")

    print("[DEBUG] Loading face detector model from:", model_path)
    BaseOptions         = mp_tasks.BaseOptions
    FaceDetectorOptions = vision.FaceDetectorOptions
    VisionRunningMode   = vision.RunningMode
    fd_opts = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=0.3
    )
    face_detector = vision.FaceDetector.create_from_options(fd_opts)

    sol_face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.3
    )   
    print("[DEBUG] Loading pose landmarker model from path...")
    pose_landmarker = create_pose_landmarker(pose_path)
    resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

    rgb_buffer, resp_buffer = [], []

    plt.ion()
    fig, (ax_rppg, ax_resp) = plt.subplots(2, 1, figsize=(6, 6))
    ax_rppg.set_title("rPPG (filtered)"); ax_resp.set_title("Respirasi")
    ax_rppg.set_xlabel("Frame");      ax_resp.set_xlabel("Frame")
    ax_rppg.grid(True);               ax_resp.grid(True)

    frame_idx = 0
    initialized = False

    try:
        show_countdown_overlay(cap, duration=5)
        while True:
            ret, frame = cap.read()
            print(f"[DEBUG] Frame {frame_idx}: ret={ret}")
            if not ret:
                break
            frame = cv2.resize(frame, (960, 720)) 

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            res = face_detector.detect_for_video(mp_img, timestamp_ms)
            num_det = len(res.detections)
            print(f"[DEBUG] Tasks detections: {num_det}")

            if num_det == 0:
                sol = sol_face_det.process(img_rgb)
                if sol.detections:
                    d = sol.detections[0].location_data.relative_bounding_box
                    fh, fw = frame.shape[:2]
                    x = int(d.xmin * fw)
                    y = int(d.ymin * fh)
                    W = int(d.width * fw)
                    H = int(d.height * fh)

                    class Dummy: pass
                    det = Dummy()
                    det.bounding_box = Dummy()
                    det.bounding_box.origin_x = x
                    det.bounding_box.origin_y = y
                    det.bounding_box.width    = W
                    det.bounding_box.height   = H
                    res.detections = [det]
                    num_det = 1
                    print("[DEBUG] Fallback Sol API detection used")

            if num_det == 0:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                frame_idx += 1
                continue

            det  = res.detections[0]
            bbox = det.bounding_box
            x, y = int(bbox.origin_x), int(bbox.origin_y)
            W, H = int(bbox.width),    int(bbox.height)

            cx, cy = x + W//2, y + H//2
            R = min(W, H) // 3
            l, r = max(0, cx - R), min(frame.shape[1], cx + R)
            t, b = max(0, cy - R), min(frame.shape[0], cy + R)

            if r - l <= 0 or b - t <= 0:
                print("[DEBUG] Invalid ROI size, skipping.")
                frame_idx += 1
                continue

            roi = frame[t:b, l:r]
            cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 2)

            mean_bgr = cv2.mean(roi)[:3]
            rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

            if not initialized:
                try:
                    resp_tracker.initialize(frame, timestamp_ms=timestamp_ms)
                    initialized = True
                    print("[DEBUG] RespTracker initialized.")
                except Exception as e:
                    print("[DEBUG] RespTracker init failed:", e)

            if initialized:
                try:
                    # Update Optical Flow untuk sinyal respirasi
                    resp_y = resp_tracker.update(frame)
                    resp_buffer.append(resp_y)

                    # Update ulang titik bahu dari pose terbaru
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    res = pose_landmarker.detect_for_video(mp_img, timestamp_ms=timestamp_ms)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks[0]
                        ls, rs = lm[11], lm[12]
                        h, w = frame.shape[:2]
                        resp_tracker.shoulder_pts = [(int(ls.x * w), int(ls.y * h)), (int(rs.x * w), int(rs.y * h))]

                    # Gambar titik bahu terbaru
                    if resp_tracker.shoulder_pts:
                        for pt in resp_tracker.shoulder_pts:
                            cv2.circle(frame, pt, radius=5, color=(0, 0, 255), thickness=-1)

                except Exception as e:
                    print("[DEBUG] RespTracker update failed:", e)


            # Tambahkan teks instruksi
            cv2.putText(frame, "Tekan Q untuk selesai", (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if frame_idx % 10 == 0 and len(rgb_buffer) >= WIN_POS:
                rgb_arr = np.array(rgb_buffer).T
                rppg_sig = extract_rppg(rgb_arr, fps=FPS, lowcut=LOW_RPPG, highcut=HIGH_RPPG)
                resp_sig = bandpass_filter(np.array(resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS)

                ax_rppg.cla(); ax_resp.cla()

                ax_rppg.plot(rppg_sig, color='blue', label='rPPG')
                ax_rppg.set_title("Sinyal rPPG (detak jantung)")
                ax_rppg.set_xlabel("Frame ke-"); ax_rppg.set_ylabel("Amplitudo")
                ax_rppg.legend(); ax_rppg.grid(True)

                ax_resp.plot(resp_sig, color='green', label='Respirasi')
                ax_resp.set_title("Sinyal Respirasi (gerak bahu)")
                ax_resp.set_xlabel("Frame ke-"); ax_resp.set_ylabel("Posisi Y (px)")
                ax_resp.legend(); ax_resp.grid(True)

                fig.tight_layout()
                fig.canvas.draw(); plt.pause(0.001)

    finally:
        print("[DEBUG] Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        sol_face_det.close()
        pose_landmarker.close()

if __name__ == "__main__":
    main()
