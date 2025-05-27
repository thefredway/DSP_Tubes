import tkinter as tk
from tkinter import messagebox
from threading import Thread
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os
import csv
import time
from datetime import datetime
from rppg_utils import extract_rppg
from resp_utils import create_pose_landmarker, RespTracker
from filter_utils import bandpass_filter

FPS = 30.0
LOW_RPPG, HIGH_RPPG = 0.8, 2.5
LOW_RESP, HIGH_RESP = 0.1, 0.7

class GUIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("DSP GUI - rPPG & Respirasi")
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.pack()

        self.btn_start = tk.Button(master, text="Mulai Rekam", command=self.start_recording)
        self.btn_start.pack(pady=10)

        self.btn_waveform = tk.Button(master, text="Lihat Waveform", command=self.plot_waveform)
        self.btn_waveform.pack(pady=5)

        self.cap = None
        self.frame = None
        self.running = False
        self.rgb_buffer = []
        self.resp_buffer = []

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
        if self.running:
            self.master.after(10, self.update_frame)

    def countdown(self, duration=5):
        for i in range(duration, 0, -1):
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.putText(frame, f"Mulai dalam {i}", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.master.update()
            time.sleep(1)

    def start_recording(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka webcam.")
            return

        self.running = True
        self.update_frame()

        # Countdown
        self.countdown(5)

        # Load model
        pose_path = os.path.join("models", "pose_landmarker.task")
        pose_landmarker = create_pose_landmarker(pose_path)
        resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

        self.rgb_buffer.clear()
        self.resp_buffer.clear()

        initialized = False
        frame_idx = 0

        while frame_idx < 300:  # capture 10 seconds at 30fps
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))

            # ROI wajah
            cx, cy, R = 320, 240, 80
            l, r = max(0, cx - R), min(frame.shape[1], cx + R)
            t, b = max(0, cy - R), min(frame.shape[0], cy + R)
            roi = frame[t:b, l:r]
            mean_bgr = cv2.mean(roi)[:3]
            self.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

            # Tracking respirasi
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

            frame_idx += 1
            self.canvas.update()

        self.running = False
        self.cap.release()

        # Simpan CSV
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("rppg_data", exist_ok=True)
        np.savetxt(f"rppg_data/rppg_{now}.csv", np.array(self.rgb_buffer), delimiter=",")
        np.savetxt(f"rppg_data/resp_{now}.csv", np.array(self.resp_buffer), delimiter=",")

        self.plot_waveform()

    def plot_waveform(self):
        if not self.rgb_buffer or not self.resp_buffer:
            messagebox.showwarning("Peringatan", "Belum ada data yang direkam.")
            return

        rgb_arr = np.array(self.rgb_buffer).T
        rppg = extract_rppg(rgb_arr, fps=FPS, lowcut=LOW_RPPG, highcut=HIGH_RPPG)
        resp = bandpass_filter(np.array(self.resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(rppg, color='blue'); axs[0].set_title("rPPG Signal"); axs[0].set_xlabel("Time")
        axs[1].plot(resp, color='green'); axs[1].set_title("Respiration Signal"); axs[1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
