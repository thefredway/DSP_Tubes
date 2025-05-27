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
        self.master.attributes("-fullscreen", True)

        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === Left Panel ===
        self.canvas = tk.Canvas(self.left_frame, width=960, height=720)
        self.canvas.pack()

        self.countdown_label = tk.Label(self.left_frame, text="", font=("Arial", 48), fg="red")
        self.countdown_label.pack(pady=10)

        self.duration_label = tk.Label(self.left_frame, text="Durasi (detik):")
        self.duration_label.pack()
        self.duration_entry = tk.Entry(self.left_frame)
        self.duration_entry.insert(0, "10")
        self.duration_entry.pack()

        self.btn_start = tk.Button(self.left_frame, text="Mulai Rekam", command=self.start_recording_thread)
        self.btn_start.pack(pady=10)

        self.btn_exit = tk.Button(self.left_frame, text="Keluar", command=self.exit_program)
        self.btn_exit.pack(pady=10)

        # === Right Panel ===
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cap = None
        self.running = False
        self.rgb_buffer = []
        self.resp_buffer = []
        self.now = ""

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (960, 720))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
        if self.running:
            self.master.after(10, self.update_frame)

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
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka webcam.")
            return

        self.running = True
        self.update_frame()
        self.countdown(5)

        pose_path = os.path.join("models", "pose_landmarker.task")
        pose_landmarker = create_pose_landmarker(pose_path)
        resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

        self.rgb_buffer.clear()
        self.resp_buffer.clear()
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

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(rppg, color='blue', label="rPPG")
        self.ax1.plot(peaks_rppg, rppg[peaks_rppg], 'rx')
        self.ax1.set_title(f"rPPG Signal\nBPM ≈ {bpm:.1f}")
        self.ax1.set_xlabel("Time"); self.ax1.legend(); self.ax1.grid(True)

        self.ax2.plot(resp, color='green', label="Respiration")
        self.ax2.plot(peaks_resp, resp[peaks_resp], 'rx')
        self.ax2.set_title(f"Respiration Signal\nBR ≈ {br:.1f} bpm")
        self.ax2.set_xlabel("Time"); self.ax2.legend(); self.ax2.grid(True)

        self.canvas_plot.draw()

    def exit_program(self):
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
