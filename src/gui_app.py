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
from cso import cat_swarm_optimize, bandpass_and_eval

FPS = 30.0
DEFAULT_LOW_RPPG = 0.8
DEFAULT_HIGH_RPPG = 2.5
DEFAULT_ORDER = 4
LOW_RESP, HIGH_RESP = 0.1, 0.7

class GUIApp:
    """
    Kelas utama GUI berbasis Tkinter untuk pemrosesan sinyal rPPG dan respirasi secara real-time.
    Menggabungkan input webcam, deteksi wajah/bahu, filter, dan plotting sinyal.

    Parameter:
    - master : objek Tkinter utama (tk.Tk)
    """
    def __init__(self, master):
        """
        Inisialisasi komponen GUI, kamera, dan plotting.
        """
        self.master = master
        self.master.title("DSP GUI - rPPG & Respirasi")
        self.master.state('zoomed')
        self.master.bind("<Escape>", lambda e: self.exit_program())

        # === Layout utama: kiri video, kanan plot ===
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(main_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        # === Label video ===
        self.video_label = tk.Label(self.left_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # === Label status (contoh: sedang merekam...) ===
        self.status_label = tk.Label(self.left_frame, text="", font=("Arial", 24), fg="blue")
        self.status_label.grid(row=1, column=0, pady=5)

        # === Frame kontrol: tombol dan input user ===
        self.controls = tk.Frame(self.left_frame)
        self.controls.grid(row=2, column=0, pady=10)

        # Input durasi
        tk.Label(self.controls, text="Durasi (detik):").grid(row=0, column=0)
        self.duration_entry = tk.Entry(self.controls, width=5)
        self.duration_entry.insert(0, "10")
        self.duration_entry.grid(row=0, column=1)

        # Tombol mulai rekam
        self.btn_start = tk.Button(self.controls, text="Mulai Rekam", command=self.start_recording_thread)
        self.btn_start.grid(row=0, column=2, padx=10)

        # Label BPM dan BR
        self.bpm_label = tk.Label(self.controls, text="BPM: -")
        self.bpm_label.grid(row=0, column=3, padx=5)
        self.br_label = tk.Label(self.controls, text="BR: -")
        self.br_label.grid(row=0, column=4, padx=5)

        # Tombol keluar
        self.btn_exit = tk.Button(self.controls, text="❌ Keluar", command=self.exit_program, bg="red", fg="white")
        self.btn_exit.grid(row=0, column=5, padx=10)

        # Parameter rPPG
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

        # Tombol optimasi dan bantuan
        tk.Button(self.controls, text="🔍 Optimasi Parameter rPPG", command=self.run_filter_optimization).grid(row=2, column=0, pady=5)
        tk.Button(self.controls, text="🔍 Optimasi Parameter Respirasi", command=self.run_resp_optimization).grid(row=2, column=1, pady=5)
        tk.Button(self.controls, text="🆘 Help", command=self.show_help).grid(row=2, column=2, pady=5)

        # Output untuk parameter respirasi
        tk.Label(self.controls, text="Resp Low (Hz):").grid(row=3, column=0)
        self.low_resp_label = tk.Label(self.controls, text=f"{LOW_RESP:.2f}")
        self.low_resp_label.grid(row=3, column=1)
        tk.Label(self.controls, text="Resp High (Hz):").grid(row=3, column=2)
        self.high_resp_label = tk.Label(self.controls, text=f"{HIGH_RESP:.2f}")
        self.high_resp_label.grid(row=3, column=3)

        # === Grafik rPPG dan respirasi (matplotlib embedded) ===
        self.figure = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax_rppg = self.figure.add_subplot(211)
        self.ax_resp = self.figure.add_subplot(212)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === Inisialisasi variabel tracking dan buffer ===
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.blink = False
        self.blink_id = None
        self.rgb_buffer = deque(maxlen=int(FPS * 30))
        self.resp_buffer = deque(maxlen=int(FPS * 30))
        self.last_update_time = time.time()
        self.update_video_frame()

    def blink_status(self):
        """
        Menampilkan tulisan "Sedang Merekam..." berkedip setiap 0.5 detik.
        Dipanggil secara rekursif dengan `after`.
        """
        if not self.running:
            self.status_label.config(text="")
            return
        self.blink = not self.blink
        self.status_label.config(text="Sedang Merekam..." if self.blink else "")
        self.blink_id = self.master.after(500, self.blink_status)

    def update_video_frame(self):
        """
        Update frame video dari webcam ke Tkinter setiap 10 ms (real-time).
        """
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
        """
        Memulai thread baru untuk proses countdown dan perekaman.
        Menghindari freeze GUI saat countdown.
        """
        Thread(target=self.start_with_countdown).start()

    def start_with_countdown(self):
        """
        Countdown 5 detik sebelum memulai proses rekaman.
        Setelah selesai countdown, mulai rekaman sinyal.
        """
        self.running = False
        for i in range(5, 0, -1):
            self.master.after(0, lambda x=i: self.status_label.config(text=f"Mulai dalam {x}..."))
            time.sleep(1)
        self.master.after(0, lambda: self.status_label.config(text="Sedang Merekam..."))
        self.running = True
        self.master.after(0, self.blink_status)
        self.start_recording()

    def start_recording(self):
        """
        Melakukan perekaman sinyal rPPG dan respirasi dari webcam selama durasi tertentu.
        Hasil disimpan ke file CSV.
        """
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
        
        # Inisialisasi pose landmark dan tracker respirasi
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

            # Ambil ROI wajah tengah untuk rPPG
            roi = frame[h//3:h//3+120, w//2-60:w//2+60]
            mean_bgr = cv2.mean(roi)[:3]
            self.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

            # Inisialisasi tracking bahu
            if not initialized:
                try:
                    resp_tracker.initialize(frame, timestamp_ms=frame_idx * 33)
                    initialized = True
                except Exception:
                    pass

            # Tracking respirasi dari optical flow
            if initialized:
                try:
                    resp_y = resp_tracker.update(frame)
                    self.resp_buffer.append(resp_y)
                except Exception:
                    pass

            # Update grafik real-time setiap 2 detik
            if time.time() - self.last_update_time > 2:
                self.master.after(0, self.update_realtime_plot)
                self.last_update_time = time.time()

            frame_idx += 1
            self.master.update()

        self.running = False
        if self.blink_id:
            self.master.after(0, lambda: self.master.after_cancel(self.blink_id))
        self.master.after(0, lambda: self.status_label.config(text=""))

        # Simpan data hasil rekaman ke file CSV
        os.makedirs("rppg_data", exist_ok=True)
        rppg_path = f"rppg_data/rppg_{now}.csv"
        resp_path = f"rppg_data/resp_{now}.csv"
        np.savetxt(rppg_path, np.array(self.rgb_buffer), delimiter=",")
        np.savetxt(resp_path, np.array(self.resp_buffer), delimiter=",")
        self.master.after(0, lambda: messagebox.showinfo("Rekaman Selesai", f"Rekaman selesai dan disimpan di:\n{rppg_path}"))
        self.master.after(0, self.update_realtime_plot)

    def run_filter_optimization(self):
        """
        Melakukan optimasi parameter filter rPPG menggunakan algoritma Cat Swarm Optimization (CSO).
        Parameter yang dioptimasi: lowcut, highcut, dan order filter.
        """
        self.status_label.config(text="⚠️ Harap diam saat optimasi filter...")
        self.master.update()
        time.sleep(1.5)
        self.status_label.config(text="⏳ Sedang mengoptimasi filter...")
        self.master.update()

        try:
            rgb_arr = np.array(self.rgb_buffer).T
            if rgb_arr.shape[1] < FPS * 3:
                messagebox.showwarning("Buffer Kosong", "Sinyal belum cukup untuk optimasi.")
                return
        except Exception:
            messagebox.showerror("Error", "Gagal mengakses buffer.")
            return

        # Ekstraksi rPPG awal sebagai sinyal dasar
        signal = extract_rppg(rgb_arr, fps=FPS, lowcut=0.8, highcut=2.5)
        fs = FPS

        def obj(x):
            return bandpass_and_eval(signal, fs, bandpass_filter, x)

        bounds = [(0.6, 1.2), (2.0, 3.0), (2, 8.01)]

        best_param, best_score = cat_swarm_optimize(
            objective_func=obj,
            bounds=bounds,
            n_cats=12,
            max_iter=25
        )

        low, high, order = best_param
        self.low_rppg_entry.delete(0, tk.END)
        self.low_rppg_entry.insert(0, f"{low:.3f}")
        self.high_rppg_entry.delete(0, tk.END)
        self.high_rppg_entry.insert(0, f"{high:.3f}")
        self.order_entry.delete(0, tk.END)
        self.order_entry.insert(0, f"{int(order)}")

        self.status_label.config(text="✅ Optimasi selesai. Parameter terbaik diterapkan.")
        self.master.after(3000, lambda: self.status_label.config(text=""))
        self.update_realtime_plot()

    def run_resp_optimization(self):
        """
        Melakukan optimasi parameter filter sinyal respirasi menggunakan CSO.
        Parameter yang dioptimasi: lowcut dan highcut respirasi.
        """
        global LOW_RESP, HIGH_RESP
        self.status_label.config(text="⚠️ Harap diam saat optimasi respirasi...")
        self.master.update()
        time.sleep(1.5)
        self.status_label.config(text="⏳ Sedang mengoptimasi respirasi...")
        self.master.update()

        signal = np.array(self.resp_buffer)
        fs = FPS

        def obj(x):
            return bandpass_and_eval(signal, fs, bandpass_filter, x)

        bounds = [(0.05, 0.4), (0.5, 0.9), (2, 8.01)]
        best_param, _ = cat_swarm_optimize(obj, bounds, n_cats=12, max_iter=25)

        LOW_RESP, HIGH_RESP, _ = best_param
        self.low_resp_label.config(text=f"{LOW_RESP:.2f}")
        self.high_resp_label.config(text=f"{HIGH_RESP:.2f}")        
        self.status_label.config(text="✅ Optimasi respirasi selesai.")
        self.master.after(3000, lambda: self.status_label.config(text=""))
        self.update_realtime_plot()

    def show_help(self):
        """
        Menampilkan panduan penggunaan aplikasi dalam bentuk pop-up message.
        Menjelaskan urutan rekaman dan optimasi.
        """
        help_text = (
            "🎬 Urutan Disarankan: Kalibrasi & Optimasi\n\n"
            "1️⃣ Rekaman Kalibrasi (10 detik)\n"
            "- Klik 'Mulai Rekam' untuk merekam wajah dan bahu\n"
            "- Jangan banyak gerak, pastikan pencahayaan cukup\n\n"
            "2️⃣ Optimasi Parameter\n"
            "- Klik '🔍 Optimasi Parameter rPPG' → parameter wajah\n"
            "- Klik '🔍 Optimasi Parameter Respirasi' → parameter bahu\n"
            "- Sistem mencari parameter terbaik untuk filter\n\n"            
            "3️⃣ Rekaman Utama\n"
            "- Klik 'Mulai Rekam' lagi sesuai dengan waktu yang diinginkan\n"
            "- Menggunakan parameter hasil optimasi\n"
            "- Grafik dan estimasi BPM/BR lebih stabil"
        )
        messagebox.showinfo("Panduan Penggunaan", help_text)

    def update_realtime_plot(self):
        """
        Memperbarui grafik matplotlib dengan sinyal rPPG dan respirasi terbaru.
        Menampilkan titik puncak dan menghitung estimasi BPM dan BR.
        """
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
        """
        Menghentikan webcam dan menutup GUI.
        Dipanggil saat klik tombol ❌ atau tekan tombol Escape.
        """
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
