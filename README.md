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
| `opencv-python`                      | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung.                 |
| `numpy`                | 	Digunakan untuk manipulasi data array.|        
| `mediapipe` | Digunakan untuk deteksi wajah dan landmark tubuh secara real-time menggunakan model.  |
|`scipy` | Digunakan untuk menerapkan filter sinyal band-pass. |
|`matplotlib` | 	Digunakan untuk memvisualisasikan sinyal rPPG (detak jantung) dan respirasi. | 


---
