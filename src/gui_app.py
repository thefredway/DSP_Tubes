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
import ctypes

from rppg_utils import extract_rppg
from resp_utils import create_pose_landmarker, RespTracker
from filter_utils import bandpass_filter

FPS = 30.0
DEFAULT_LOW_RPPG = 0.8
DEFAULT_HIGH_RPPG = 2.5
DEFAULT_ORDER = 4
LOW_RESP, HIGH_RESP = 0.1, 0.7

class GUIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("DSP GUI - rPPG & Respirasi")
        self.master.state('zoomed')
        self.master.bind("<Escape>", lambda e: self.exit_program())

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(main_frame)
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
        self.duration_entry = tk.Entry(self.controls, width=5)
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

        tk.Label(self.controls, text="rPPG Low:").grid(row=1, column=0)
        self.low_rppg_entry = tk.Entry(self.controls, width=5)
        self.low_rppg_entry.insert(0, str(DEFAULT_LOW_RPPG))
        self.low_rppg_entry.grid(row=1, column=1)

        tk.Label(self.controls, text="High:").grid(row=1, column=2)
        self.high_rppg_entry = tk.Entry(self.controls, width=5)
        self.high_rppg_entry.insert(0, str(DEFAULT_HIGH_RPPG))
        self.high_rppg_entry.grid(row=1, column=3)

        tk.Label(self.controls, text="Order:").grid(row=1, column=4)
        self.order_entry = tk.Entry(self.controls, width=5)
        self.order_entry.insert(0, str(DEFAULT_ORDER))
        self.order_entry.grid(row=1, column=5)

        self.figure = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax_rppg = self.figure.add_subplot(211)
        self.ax_resp = self.figure.add_subplot(212)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.blink = False
        self.blink_id = None
        self.rgb_buffer = deque(maxlen=int(FPS * 30))
        self.resp_buffer = deque(maxlen=int(FPS * 30))
        self.last_update_time = time.time()
        self.update_video_frame()

    def blink_status(self):
        if not self.running:
            self.status_label.config(text="")
            return
        self.blink = not self.blink
        self.status_label.config(text="Sedang Merekam..." if self.blink else "")
        self.blink_id = self.master.after(500, self.blink_status)

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
        Thread(target=self.start_with_countdown).start()

    def start_with_countdown(self):
        self.running = False
        for i in range(5, 0, -1):
            self.master.after(0, lambda x=i: self.status_label.config(text=f"Mulai dalam {x}..."))
            time.sleep(1)
        self.master.after(0, lambda: self.status_label.config(text="Sedang Merekam..."))
        self.running = True
        self.master.after(0, self.blink_status)
        self.start_recording()

    def start_recording(self):
        try:
            duration_sec = int(self.duration_entry.get())
            if duration_sec <= 0:
                raise ValueError
        except ValueError:
            self.master.after(0, lambda: messagebox.showerror("Input Error", "Durasi harus angka > 0"))
            return

        frame_limit = int(duration_sec * FPS)
        if not self.cap.isOpened():
            self.master.after(0, lambda: messagebox.showerror("Error", "Tidak dapat membuka webcam."))
            return

        pose_path = os.path.join("models", "pose_landmarker.task")
        pose_landmarker = create_pose_landmarker(pose_path)
        resp_tracker = RespTracker(pose_landmarker, x_size=150, y_size=120, shift_x=0, shift_y=40)

        self.rgb_buffer.clear()
        self.resp_buffer.clear()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        initialized = False
        frame_idx = 0

        while frame_idx < frame_limit:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (960, 720))
            h, w = frame.shape[:2]
            roi = frame[h//3:h//3+120, w//2-60:w//2+60]
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
                self.master.after(0, self.update_realtime_plot)
                self.last_update_time = time.time()

            frame_idx += 1
            self.master.update()

        self.running = False
        if self.blink_id:
            self.master.after(0, lambda: self.master.after_cancel(self.blink_id))
        self.master.after(0, lambda: self.status_label.config(text=""))

        os.makedirs("rppg_data", exist_ok=True)
        rppg_path = f"rppg_data/rppg_{now}.csv"
        resp_path = f"rppg_data/resp_{now}.csv"
        np.savetxt(rppg_path, np.array(self.rgb_buffer), delimiter=",")
        np.savetxt(resp_path, np.array(self.resp_buffer), delimiter=",")
        self.master.after(0, lambda: messagebox.showinfo("Rekaman Selesai", f"Rekaman selesai dan disimpan di:\n{rppg_path}"))
        self.master.after(0, self.update_realtime_plot)

    def update_realtime_plot(self):
        if len(self.rgb_buffer) < FPS * 3:
            return
        rgb_arr = np.array(self.rgb_buffer).T
        try:
            low_rppg = float(self.low_rppg_entry.get())
            high_rppg = float(self.high_rppg_entry.get())
            order = int(self.order_entry.get())
        except ValueError:
            low_rppg, high_rppg, order = DEFAULT_LOW_RPPG, DEFAULT_HIGH_RPPG, DEFAULT_ORDER

        rppg = extract_rppg(rgb_arr, fps=FPS, lowcut=low_rppg, highcut=high_rppg)
        resp = bandpass_filter(np.array(self.resp_buffer), LOW_RESP, HIGH_RESP, fs=FPS)

        peaks_rppg, _ = find_peaks(rppg, distance=FPS // 2)
        peaks_resp, _ = find_peaks(resp, distance=FPS * 2)

        duration_sec = len(rppg) / FPS
        time_axis = np.linspace(0, duration_sec, len(rppg))
        bpm = len(peaks_rppg) * (60 / duration_sec)
        br = len(peaks_resp) * (60 / duration_sec)

        self.bpm_label.config(text=f"BPM: {bpm:.1f}")
        self.br_label.config(text=f"BR: {br:.1f}")

        self.ax_rppg.clear()
        self.ax_resp.clear()

        self.ax_rppg.plot(time_axis, rppg, color='blue', label="rPPG")
        self.ax_rppg.plot(time_axis[peaks_rppg], rppg[peaks_rppg], 'rx', label="Peak")
        self.ax_rppg.set_title("rPPG Signal (Remote PPG)", fontsize=12)
        self.ax_rppg.set_xlabel("Time (seconds)", fontsize=10)
        self.ax_rppg.set_ylabel("Amplitude", fontsize=10)
        self.ax_rppg.legend(fontsize=9)
        self.ax_rppg.grid(True)
        self.ax_rppg.tick_params(axis='both', labelsize=8)

        self.ax_resp.plot(time_axis[:len(resp)], resp, color='green', label="Respiration")
        self.ax_resp.plot(time_axis[peaks_resp], resp[peaks_resp], 'rx', label="Peak")
        self.ax_resp.set_title("Respiration Signal (Chest Movement)", fontsize=12)
        self.ax_resp.set_xlabel("Time (seconds)", fontsize=10)
        self.ax_resp.set_ylabel("Displacement (px)", fontsize=10)
        self.ax_resp.legend(fontsize=9)
        self.ax_resp.grid(True)
        self.ax_resp.tick_params(axis='both', labelsize=8)

        self.figure.tight_layout(pad=3.0)
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
