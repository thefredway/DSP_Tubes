import tkinter as tk
from tkinter import messagebox
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

FPS = 30.0
LOW_RPPG, HIGH_RPPG = 0.8, 2.5
LOW_RESP, HIGH_RESP = 0.1, 0.7

class GUIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("DSP GUI - rPPG & Respirasi")
        self.master.geometry("1600x900")
        self.master.bind("<Escape>", lambda e: self.exit_program())

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(main_frame, width=960)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(self.left_frame)
        self.video_label.pack()

        self.top_controls = tk.Frame(self.left_frame)
        self.top_controls.pack(pady=5)
        self.countdown_label = tk.Label(self.top_controls, text="", font=("Arial", 32), fg="red")
        self.countdown_label.pack()

        self.controls = tk.Frame(self.left_frame)
        self.controls.pack(pady=10)

        self.duration_label = tk.Label(self.controls, text="Durasi (detik):")
        self.duration_label.grid(row=0, column=0)
        self.duration_entry = tk.Entry(self.controls)
        self.duration_entry.insert(0, "10")
        self.duration_entry.grid(row=0, column=1)

        self.btn_start = tk.Button(self.controls, text="Mulai Rekam", command=self.start_recording_thread)
        self.btn_start.grid(row=0, column=2, padx=10)

        self.btn_exit = tk.Button(self.controls, text="❌ Keluar", command=self.exit_program, bg="red", fg="white")
        self.btn_exit.grid(row=0, column=4, padx=10)

        # === Dual Subplot ===
        self.figure = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax_rppg = self.figure.add_subplot(211)
        self.ax_resp = self.figure.add_subplot(212)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.rgb_buffer = []
        self.resp_buffer = []
        self.r_signal = deque(maxlen=int(FPS*10))
        self.g_signal = deque(maxlen=int(FPS*10))
        self.b_signal = deque(maxlen=int(FPS*10))

        self.update_video_frame()

    def update_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (960, 720))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.video_label.config(image=img)
                self.video_label.image = img
        self.master.after(10, self.update_video_frame)

    def countdown(self, seconds=5):
        for i in range(seconds, 0, -1):
            self.countdown_label.config(text=str(i))
            self.master.update()
            time.sleep(1)
        self.countdown_label.config(text="")

    def start_recording_thread(self):
        thread = Thread(target=self.start_recording)
        thread.start()

    def start_recording(self):
        try:
            duration_sec = int(self.duration_entry.get())
            if duration_sec <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Durasi harus berupa angka > 0")
            return

        frame_limit = int(duration_sec * FPS)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka webcam.")
            return

        self.running = True
        self.countdown(5)

        pose_path = os.path.join("models", "pose_landmarker.task")
        pose_landmarker = create_pose_landmarker(pose_path)
        resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

        self.rgb_buffer.clear()
        self.resp_buffer.clear()
        self.r_signal.clear()
        self.g_signal.clear()
        self.b_signal.clear()
        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")

        initialized = False
        frame_idx = 0

        while frame_idx < frame_limit:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.resize(frame, (960, 720))

            cx, cy, R = 480, 360, 100
            l, r = max(0, cx - R), min(frame.shape[1], cx + R)
            t, b = max(0, cy - R), min(frame.shape[0], cy + R)
            roi = frame[t:b, l:r]
            mean_bgr = cv2.mean(roi)[:3]
            self.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
            self.r_signal.append(mean_bgr[2])
            self.g_signal.append(mean_bgr[1])
            self.b_signal.append(mean_bgr[0])

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
            self.master.update()

        self.running = False
        os.makedirs("rppg_data", exist_ok=True)
        np.savetxt(f"rppg_data/rppg_{self.now}.csv", np.array(self.rgb_buffer), delimiter=",")
        np.savetxt(f"rppg_data/resp_{self.now}.csv", np.array(self.resp_buffer), delimiter=",")

        self.plot_waveform()

    def plot_waveform(self):
        if not self.rgb_buffer or not self.resp_buffer:
            messagebox.showwarning("Peringatan", "Belum ada data yang direkam.")
            return

        rgb_arr = np.array(self.rgb_buffer).T
        rppg = extract_rppg(rgb_arr, fps=FPS, lowcut=LOW_RPPG, highcut=HIGH_RPPG)
        resp = bandpass_filter(np.array(self.resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS)

        peaks_rppg, _ = find_peaks(rppg, distance=FPS//2)
        peaks_resp, _ = find_peaks(resp, distance=FPS*2)

        duration_sec = len(rppg) / FPS
        bpm = len(peaks_rppg) * (60 / duration_sec)
        br = len(peaks_resp) * (60 / duration_sec)

        self.ax_rppg.clear()
        self.ax_resp.clear()

        self.ax_rppg.plot(rppg, color='blue', label="rPPG")
        self.ax_rppg.plot(peaks_rppg, rppg[peaks_rppg], 'rx')
        self.ax_rppg.set_title(f"rPPG Signal\nBPM ≈ {bpm:.1f}")
        self.ax_rppg.set_xlabel("Frame")
        self.ax_rppg.legend()
        self.ax_rppg.grid(True)

        self.ax_resp.plot(resp, color='green', label="Respiration")
        self.ax_resp.plot(peaks_resp, resp[peaks_resp], 'rx')
        self.ax_resp.set_title(f"Respiration Signal\nBR ≈ {br:.1f} bpm")
        self.ax_resp.set_xlabel("Frame")
        self.ax_resp.legend()
        self.ax_resp.grid(True)

        self.canvas_plot.draw()

    def exit_program(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
