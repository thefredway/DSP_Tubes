# DSP_Tubes
Dibuat untuk menyelesaikan tugas besar mata kuliah Pengolahan Sinyal Digital IF3024 Institut Teknologi Sumatera


## **Deskripsi Tugas Besar**
Proyek ini merupakan tugas akhir dari mata kuliah Pengolahan Sinyal Digital IF(3024) yang dilakukan untuk mengekstraksi sinyal respirasi dan sinyal _remote-photoplethysmography_ (rPPG) secara _real-time_ menggunakan video dari webcam.

Sinyal respirasi diperoleh dengan memanfaatkan _pose-landmarker_ MediaPipe untuk mendeteksi gerakan bahu saat bernapas. Sedangkan sinyal rPPG dihasilkan dengan menggunakan _face-detector _MediaPipe dan algoritma _Plane Orthogonal-to-Skin_ (POS), yang menganalisis perubahan warna pada wajah untuk menghitung detak jantung tanpa kontak langsung.


## **Anggota Kelompok**

| Nama              | NIM       | GitHub                                                                 |
|-------------------|-----------|------------------------------------------------------------------------|
| Freddy Harahap | 1221400   | [@thefredway](https://github.com/thefredway)                           |
| Angga Dwi Arthur        | 122140144   | [@dwiarthurrevangga](https://github.com/dwiarthurrevangga)             |
| Siti Nur Aarifah      | 122450006 | [@ifaaja11](https://github.com/ifaaja11)                               |


---
## **Library**   

Beberapa library Python yang digunakan dalam Tugas Besar ini, beserta fungsinya:

| **Library**                | **Fungsi**                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
|`Tkinter`|  Digunakan untuk membangun antarmuka pengguna grafis (GUI) berbasis Python seperti tampilan video, tombol kontrol, dan grafik sinyal secara _real-time._|
| `opencv-python`                      | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung.                 |
| `numpy`                | 	Digunakan untuk manipulasi data array.|        
| `mediapipe` | Digunakan untuk deteksi wajah dan landmark tubuh secara real-time menggunakan model.  |
|`scipy` | Digunakan untuk menerapkan filter sinyal band-pass. |
|`matplotlib` | 	Digunakan untuk memvisualisasikan sinyal rPPG (detak jantung) dan respirasi. | 
|`PIL (Pillow)` | Digunakan untuk konversi frmae OpenCV menjadi format yang bisa ditampilkan di Tkinter.|
|`Thereading` | Digunakan untuk menjalankan proses perekaman dan pemrosesan secara paralel agar GUI tetap responsif.|
|`Time dan Datetime`|Digunakan untuk menghitung durasi perekaman, mencatat timestamp, dan memberi nama file hasil rekaman berdasarkan waktu.|
|`os`| Digunakan untuk membuat folder dan mengatur path file hasil rekaman CSV.|
|`Ctypes`|Digunakan untuk mengatur DPI awareness agar tampilan GUI tidak buram pada layar dengan resolusi tinggi.|
|`Random`|Digunakan dalam implementasi algoritma Cat Swarm Optimizationcuntuk menginisialisasi populasi dan variasi kandidat parameter.|
|`Collections.deque`|  Digunakan untuk menyimpan buffer sinyal RGB dan respirasi secara efisien dengan batas waktu (_rolling buffer_).|

---

## How to run this

Dengan asumsi bahwa Anda sudah mempunyai environment manager seperti conda. maka buat environment baru seperti ini. Clone / fork lalu jalankan perintah ini.

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
