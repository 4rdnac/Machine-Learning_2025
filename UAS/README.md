# Laporan — Model Klasifikasi Sayur (UAS Machine Learning 2025)

## Kelompok 1
| No | Nama | NIM | Role | GitHub |
|---:|------|-----|------|--------|
| 1 | Alexander Agung Raya | 2341720040 | Backend | https://github.com/AlexanderDev2004/2341720040_ML_2025 |
| 2 | Annisa Eka Puspita | 2341720131 | ML and PCVK developer | https://github.com/annisaeka123/2341720131_ML_2025 |
| 3 | Ekya Muhammad Hasfi Fadlilurrahman | 2341720111 | Fullstack | https://github.com/ekyaaa/2341720111_ML_2025 |
| 4 | Candra Ahmad Dani | 2341720187 | Frontend | https://github.com/4rdnac/2341720187_ML_2025/ |

## 1. Gambaran Umum
Proyek ini membangun model klasifikasi citra untuk mengenali **5 jenis sayur** dari gambar sebagai bagian dari tugas UAS Machine Learning 2025. Pendekatan yang digunakan adalah **classic machine learning** dengan alur utama: **augmentasi & preprocessing/segmentasi → ekstraksi fitur (HOG + LBP) → reduksi dimensi (PCA) → klasifikasi (SVM)**. Model akhir disimpan dalam bentuk **pickle** dengan nama `model_sayur.pkl`.

## 2. Kelas yang Diklasifikasikan
Model ditujukan untuk membedakan 5 kelas berikut:
- `bunga_kol`
- `cabai`
- `kubis`
- `sawi_hijau`
- `sawi_putih`

## 3. Metode yang Digunakan
### 3.1 Augmentasi Data
Sebelum proses pelatihan, dilakukan **augmentasi data** untuk memperkaya variasi citra sehingga model lebih robust terhadap perubahan kondisi input (misalnya perbedaan pencahayaan, sudut pengambilan, dan variasi tampilan objek). Augmentasi membantu mengurangi overfitting serta meningkatkan kemampuan generalisasi model pada data yang belum pernah dilihat.

### 3.2 Segmentasi / Preprocessing
Setelah augmentasi, citra diproses untuk meningkatkan konsistensi input (misalnya penyamaan ukuran, perbaikan kualitas citra, serta pemisahan objek dari latar bila diterapkan). Tujuan tahap ini adalah membuat fitur yang diekstraksi lebih fokus pada objek sayur dan mengurangi pengaruh background.

### 3.3 Ekstraksi Fitur: HOG + LBP
Model memanfaatkan kombinasi dua jenis fitur:
- **HOG (Histogram of Oriented Gradients)** untuk menangkap pola **tepi dan bentuk** objek.
- **LBP (Local Binary Pattern)** untuk menangkap pola **tekstur lokal** pada permukaan sayur.

Fitur dari HOG dan LBP digabungkan menjadi satu vektor fitur sebagai representasi numerik citra.

### 3.4 Reduksi Dimensi: PCA
Karena jumlah fitur hasil gabungan HOG+LBP bisa besar, digunakan **PCA (Principal Component Analysis)** untuk mereduksi dimensi. Tujuannya:
- mengurangi kompleksitas,
- menekan noise,
- mempercepat proses klasifikasi,
- membantu generalisasi model.

### 3.5 Klasifikasi: SVM
Tahap akhir menggunakan **Support Vector Machine (SVM)** sebagai classifier untuk memetakan vektor fitur ke label kelas sayur. SVM dipilih karena cukup kuat untuk klasifikasi berbasis fitur numerik dan cenderung stabil pada dataset berukuran kecil–menengah.

## 4. Hasil Evaluasi
Berdasarkan pengujian, model mencapai akurasi sekitar **±72%** untuk klasifikasi 5 kelas sayur.

Ringkasan perilaku model (berdasarkan confusion matrix):
- **`sawi_hijau`** dan **`bunga_kol`** menunjukkan prediksi benar yang relatif tinggi (lebih konsisten).
- **`sawi_putih`** dan **`kubis`** masih sering keliru, terutama **terprediksi sebagai `bunga_kol`**.
- Kekeliruan ini mengindikasikan adanya kemiripan visual (tekstur/bentuk) antar kelas tertentu serta kemungkinan variasi data yang belum cukup beragam.

## 5. Kesimpulan
Model klasifikasi sayur berbasis **augmentasi → preprocessing/segmentasi → HOG+LBP → PCA → SVM** telah berhasil dibangun dan dapat mengenali 5 jenis sayur dengan performa uji sekitar **72%**. Model sudah cukup layak sebagai sistem bantu klasifikasi, namun masih memerlukan perbaikan pada kelas-kelas yang memiliki kemiripan visual tinggi.

## 6. Pekerjaan Lanjutan (Saran Pengembangan)
Beberapa langkah yang direkomendasikan untuk meningkatkan performa:
1. Menambah dan menyeimbangkan jumlah data pada kelas yang sering tertukar (terutama `kubis` dan `sawi_putih`).
2. Meningkatkan kualitas preprocessing/segmentasi agar objek lebih bersih dari background.
3. Melakukan tuning hyperparameter SVM (misalnya C, gamma, kernel) dengan GridSearch.
4. Mengembangkan strategi augmentasi (menambah variasi transformasi) agar semakin robust pada kondisi nyata.
5. Membandingkan dengan pendekatan deep learning (transfer learning CNN) sebagai baseline lanjutan.

## 7. Artefak
- Notebook eksperimen dan pelatihan: `UAS.ipynb`
- Model hasil pelatihan: `model_sayur.pkl`
- Dataset: `images.zip`
- Laporan PBL: https://docs.google.com/document/d/120G36Yh9WVoM-KtlM5EtXAmJprO80YREDKwDvgpvH3Y/edit?usp=sharing 
