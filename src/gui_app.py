import tkinter as tk
from tkinter import messagebox
import ctypes
from threading import Thread
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
from datetime import datetime
from scipy.signal import find_peaks
from collections import deque

from rppg_utils import extract_rppg
from resp_utils import create_pose_landmarker, RespTracker
from filter_utils import bandpass_filter

from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe import Image as MPImage

FPS = 30.0
LOW_RPPG, HIGH_RPPG = 0.8, 2.5
LOW_RESP, HIGH_RESP = 0.1, 0.7

class GUIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("DSP GUI - rPPG & Respirasi")
        self.master.state('zoomed')
        self.master.bind("<Escape>", lambda e: self.exit_program())

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(main_frame, width=int(screen_width * 0.6))
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(self.left_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = tk.Label(self.left_frame, text="", font=("Arial", 24), fg="blue")
        self.status_label.grid(row=1, column=0, pady=5)

        self.controls = tk.Frame(self.left_frame)
        self.controls.grid(row=2, column=0, pady=10)

        tk.Label(self.controls, text="Durasi (detik):").grid(row=0, column=0)
        self.duration_entry = tk.Entry(self.controls)
        self.duration_entry.insert(0, "10")
        self.duration_entry.grid(row=0, column=1)

        self.btn_start = tk.Button(self.controls, text="Mulai Rekam", command=self.start_recording_thread)
        self.btn_start.grid(row=0, column=2, padx=10)

        self.bpm_label = tk.Label(self.controls, text="BPM: -")
        self.bpm_label.grid(row=0, column=3, padx=5)

        self.br_label = tk.Label(self.controls, text="BR: -")
        self.br_label.grid(row=0, column=4, padx=5)

        self.btn_exit = tk.Button(self.controls, text="❌ Keluar", command=self.exit_program, bg="red", fg="white")
        self.btn_exit.grid(row=0, column=5, padx=10)

        self.figure = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax_rppg = self.figure.add_subplot(211)
        self.ax_resp = self.figure.add_subplot(212)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.rgb_buffer = deque(maxlen=int(FPS * 30))
        self.resp_buffer = deque(maxlen=int(FPS * 30))

        model_path = os.path.join("models", "blaze_face_short_range.tflite")
        fd_opts = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=0.5
        )
        self.face_detector = FaceDetector.create_from_options(fd_opts)

        self.blink_state = True
        self.blink_loop()

        self.last_update_time = time.time()
        self.update_video_frame()

    def blink_loop(self):
        if self.running:
            self.blink_state = not self.blink_state
            self.status_label.config(text="Sedang Merekam..." if self.blink_state else "")
        self.master.after(500, self.blink_loop)

    def update_video_frame(self):
        if not self.master.winfo_exists():
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (960, 720))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.video_label.config(image=img)
                self.video_label.image = img

        self.master.after(10, self.update_video_frame)

    def start_recording_thread(self):
        Thread(target=self.start_recording).start()

    def start_recording(self):
        try:
            duration_sec = int(self.duration_entry.get())
            if duration_sec <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Durasi harus berupa angka > 0")
            return

        self.status_label.config(text="Sedang Merekam...")
        frame_limit = int(duration_sec * FPS)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka webcam.")
            return

        pose_path = os.path.join("models", "pose_landmarker.task")
        pose_landmarker = create_pose_landmarker(pose_path)
        resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

        self.rgb_buffer.clear()
        self.resp_buffer.clear()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        initialized = False
        frame_idx = 0
        self.running = True

        while frame_idx < frame_limit:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (960, 720))
            roi = frame[320:440, 430:530]
            mean_bgr = cv2.mean(roi)[:3]
            self.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

            if not initialized:
                try:
                    resp_tracker.initialize(frame, timestamp_ms=frame_idx * 33)
                    initialized = True
                except Exception:
                    pass

            if initialized:
                try:
                    resp_y = resp_tracker.update(frame)
                    self.resp_buffer.append(resp_y)
                except Exception:
                    pass

            if time.time() - self.last_update_time > 2:
                self.update_realtime_plot()
                self.last_update_time = time.time()

            frame_idx += 1
            self.master.update()

        self.running = False
        self.status_label.config(text="")

        os.makedirs("rppg_data", exist_ok=True)
        rppg_path = f"rppg_data/rppg_{now}.csv"
        resp_path = f"rppg_data/resp_{now}.csv"
        np.savetxt(rppg_path, np.array(self.rgb_buffer), delimiter=",")
        np.savetxt(resp_path, np.array(self.resp_buffer), delimiter=",")

        messagebox.showinfo("Rekaman Selesai", f"Rekaman selesai dan disimpan di:\n{rppg_path}")

        self.update_realtime_plot()

    def update_realtime_plot(self):
        if len(self.rgb_buffer) < FPS * 3:
            return
        rgb_arr = np.array(self.rgb_buffer).T
        rppg = extract_rppg(rgb_arr, fps=FPS, lowcut=LOW_RPPG, highcut=HIGH_RPPG)
        resp = bandpass_filter(np.array(self.resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS)

        peaks_rppg, _ = find_peaks(rppg, distance=FPS // 2)
        peaks_resp, _ = find_peaks(resp, distance=FPS * 2)

        duration_sec = len(rppg) / FPS
        bpm = len(peaks_rppg) * (60 / duration_sec)
        br = len(peaks_resp) * (60 / duration_sec)

        self.bpm_label.config(text=f"BPM: {bpm:.1f}")
        self.br_label.config(text=f"BR: {br:.1f}")

        self.ax_rppg.clear()
        self.ax_resp.clear()

        self.ax_rppg.plot(rppg, color='blue', label="rPPG")
        self.ax_rppg.plot(peaks_rppg, rppg[peaks_rppg], 'rx')
        self.ax_rppg.set_title("rPPG Signal")
        self.ax_rppg.legend()
        self.ax_rppg.grid(True)

        self.ax_resp.plot(resp, color='green', label="Respiration")
        self.ax_resp.plot(peaks_resp, resp[peaks_resp], 'rx')
        self.ax_resp.set_title("Respiration Signal")
        self.ax_resp.legend()
        self.ax_resp.grid(True)

        self.canvas_plot.draw()

    def exit_program(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
