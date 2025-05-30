# DSP_Tubes

Dibuat untuk menyelesaikan tugas besar mata kuliah Pengolahan Sinyal Digital IF3024 Institut Teknologi Sumatera

## **Deskripsi Tugas Besar**

Proyek ini merupakan tugas akhir dari mata kuliah **Pengolahan Sinyal Digital IF(3024)** yang bertujuan untuk mengekstraksi sinyal **respirasi** dan sinyal **_remote-photoplethysmography_ (rPPG)** secara _real-time_ menggunakan video dari webcam.

Sinyal respirasi diperoleh dengan memanfaatkan **pose-landmarker** dari MediaPipe untuk mendeteksi gerakan bahu saat bernapas. Sedangkan sinyal rPPG dihasilkan dengan menggunakan **face-detector** MediaPipe dan algoritma **Plane Orthogonal-to-Skin (POS)**, yang menganalisis perubahan warna mikro pada wajah untuk menghitung detak jantung tanpa kontak langsung.

Proyek ini juga dilengkapi dengan antarmuka GUI berbasis Tkinter yang memungkinkan pengguna untuk:

- Melakukan rekaman video wajah dan bahu.
- Menampilkan sinyal rPPG dan respirasi secara real-time.
- Mengestimasi detak jantung (BPM) dan laju pernapasan (BR).
- Melakukan optimasi otomatis parameter filter menggunakan algoritma **Cat Swarm Optimization (CSO)**.

## **Anggota Kelompok**

| Nama             | NIM       | GitHub                                                     |
| ---------------- | --------- | ---------------------------------------------------------- |
| Freddy Harahap   | 122140018  | [@thefredway](https://github.com/thefredway)               |
| Dwi Arthur Revangga | 122140144 | [@dwiarthurrevangga](https://github.com/dwiarthurrevangga) |
| Siti Nur Aarifah | 122450006 | [@ifaaja11](https://github.com/ifaaja11)                   |

---

## **Library**

Beberapa library Python yang digunakan dalam Tugas Besar ini, beserta fungsinya:

| **Library** | **Fungsi** |
| --------------- | ------------------------------------------------------------------------- |
| `opencv-python` | Menangkap gambar dari webcam dan memproses gambar/video secara real-time. |
| `numpy` | Digunakan untuk manipulasi array dan pemrosesan numerik. |
| `mediapipe` | Deteksi wajah dan landmark tubuh (pose/bahu) menggunakan model MediaPipe. |
| `scipy` | Menerapkan filter bandpass untuk memproses sinyal fisiologis. |
| `matplotlib` | Visualisasi sinyal rPPG dan respirasi dalam bentuk grafik. |
| `Pillow` | Mengubah format gambar dari OpenCV ke Tkinter. |
| `tkinter` | Membuat GUI aplikasi dan integrasi grafik dengan antarmuka. |
| `threading` | Menjalankan proses perekaman secara paralel agar GUI tidak freeze. |
| `ctypes` | Menyesuaikan DPI agar tampilan GUI lebih tajam. |
| `datetime` | Menyimpan hasil perekaman dengan timestamp unik. |
| `collections` | Mengelola buffer data sinyal dengan efisien menggunakan deque. |
| `random` | Digunakan dalam proses evolusi populasi pada algoritma CSO. |

---

## **Fitur**

### 1. Live Video Capture

- Menampilkan video langsung dari webcam pengguna ke dalam GUI.
- Dilakukan pra-pemrosesan seperti resizing dan cropping ROI secara otomatis.

### 2. Countdown dan Perekaman Otomatis

- Fitur countdown 5 detik untuk persiapan pengguna sebelum proses rekaman dimulai.
- Durasi rekaman dapat diatur secara manual.
- Setelah rekaman, sinyal disimpan sebagai file `.csv`.

### 3. Ekstraksi dan Visualisasi Sinyal

- Ekstraksi sinyal rPPG dari wajah menggunakan metode **POS (Plane Orthogonal-to-Skin)**.
- Ekstraksi sinyal respirasi dari pergerakan bahu menggunakan **Lucas-Kanade Optical Flow**.
- Visualisasi sinyal secara real-time dalam grafik matplotlib yang terintegrasi dengan GUI.

### 4. Estimasi BPM dan BR

- Sistem secara otomatis mendeteksi puncak sinyal dan menghitung:
  - **BPM (Beats Per Minute)** untuk detak jantung.
  - **BR (Breathing Rate)** untuk laju napas.
- Ditampilkan secara dinamis di GUI.

### 5. Optimasi Parameter dengan Cat Swarm Optimization

- Menggunakan algoritma **Cat Swarm Optimization (CSO)** untuk mencari parameter filter optimal:
  - rPPG: optimasi lowcut, highcut, dan order.
  - Respirasi: optimasi lowcut dan highcut.
- Tujuan optimasi adalah memaksimalkan nilai **SNR (Signal-to-Noise Ratio)** dari sinyal hasil.

### 6. GUI Interaktif dan Responsif

- Dibangun dengan **Tkinter**.
- Termasuk kontrol input durasi, parameter filter, tombol optimasi, dan tombol keluar.
- Status perekaman ditampilkan dalam bentuk teks berkedip "Sedang Merekam...".

### 7. Bantuan Penggunaan

- Tombol Help pada GUI memberikan instruksi lengkap jika kebingungan untuk penggunaan program secara ideal.

---

## **Logbook**

### Minggu 1

- Membuat struktur proyek dan script utama untuk pembacaan video.
- Implementasi `rppg_utils.py` dan `resp_utils.py`.
- Menambahkan fungsi `cpu_POS` dan filtering sinyal.
- Eksperimen awal dengan perekaman dan plotting sinyal respirasi dan rPPG.

### Minggu 2

- Implementasi `bandpass_filter` dan analisis puncak sinyal.
- Debugging kualitas sinyal dan peningkatan hasil tracking bahu.
- Perekaman video dengan pulse marker dan ekspor data ke CSV.
- Penyesuaian konfigurasi untuk real-time tracking.

### Minggu 3

- Membangun GUI interaktif berbasis Tkinter.
- Menambahkan countdown, real-time video preview, dan plotting matplotlib.
- Penambahan input durasi perekaman dan penanganan error.
- GUI menampilkan sinyal dan indikator detak jantung/pernapasan.

### Minggu 4

- Integrasi modul optimasi parameter dengan GUI.
- Penambahan tombol bantuan dan penjelasan penggunaan aplikasi.
- Pembersihan kode, penambahan docstring, dan validasi akhir fungsi.
- Penyusunan file `README.md`, `requirements.txt`, dan laporan.

---

| **Library** | **Fungsi** |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
|`Tkinter`| Digunakan untuk membangun antarmuka pengguna grafis (GUI) berbasis Python seperti tampilan video, tombol kontrol, dan grafik sinyal secara _real-time._|
| `opencv-python` | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung. |
| `numpy` | Digunakan untuk manipulasi data array.|  
| `mediapipe` | Digunakan untuk deteksi wajah dan landmark tubuh secara real-time menggunakan model. |
|`scipy` | Digunakan untuk menerapkan filter sinyal band-pass. |
|`matplotlib` | Digunakan untuk memvisualisasikan sinyal rPPG (detak jantung) dan respirasi. |
|`PIL (Pillow)` | Digunakan untuk konversi frmae OpenCV menjadi format yang bisa ditampilkan di Tkinter.|
|`Thereading` | Digunakan untuk menjalankan proses perekaman dan pemrosesan secara paralel agar GUI tetap responsif.|
|`Time dan Datetime`|Digunakan untuk menghitung durasi perekaman, mencatat timestamp, dan memberi nama file hasil rekaman berdasarkan waktu.|
|`os`| Digunakan untuk membuat folder dan mengatur path file hasil rekaman CSV.|
|`Ctypes`|Digunakan untuk mengatur DPI awareness agar tampilan GUI tidak buram pada layar dengan resolusi tinggi.|
|`Random`|Digunakan dalam implementasi algoritma Cat Swarm Optimizationcuntuk menginisialisasi populasi dan variasi kandidat parameter.|
|`Collections.deque`| Digunakan untuk menyimpan buffer sinyal RGB dan respirasi secara efisien dengan batas waktu (_rolling buffer_).|


---

## How to run this

Dengan asumsi bahwa Anda sudah mempunyai environment manager seperti conda. maka buat environment baru seperti ini. Clone / fork lalu jalankan perintah ini. Pastikan memiliki Python 3.10+

```yaml
conda create -n respiration python
```

Lalu buka environment yang sudah dibuat sebelumnya dengan

```yaml
conda activate respiration
```

Jalankan perintah ini untuk menginstall library yang dibutuhkan.

```yaml
pip install -r requirements.txt
```

Lalu jalankan perintah ini untuk menjalankan program.

```yaml
python gui_app.py
```
