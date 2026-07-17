# Sistem Rekomendasi Model Cukur Rambut Berdasarkan Bentuk Wajah

Sistem rekomendasi gaya rambut real-time berbasis Python menggunakan deteksi titik wajah (Face Mesh), ekstraksi fitur rasio wajah geometris, klasifikasi bentuk wajah menggunakan algoritma Random Forest, dan mesin rekomendasi berbasis aturan konten.

Aplikasi ini memproses feed video dari webcam secara real-time dan memberikan rekomendasi model potongan rambut yang paling sesuai untuk bentuk wajah Anda.

---

## Fitur Utama
- **Real-Time Widescreen HD Feed**: Mengambil rekaman webcam dalam rasio asli 16:9 HD (`1280x720`) untuk menjaga akurasi proporsi wajah agar tidak terdistorsi/melar kesamping.
- **Default Fullscreen & Interactive Toggle**: Aplikasi terbuka otomatis dalam mode fullscreen penuh layar, dengan kontrol pintasan tombol **`[F]`** untuk beralih ke mode jendela (*windowed*).
- **Landmark Face Mesh**: Menggunakan Google MediaPipe Face Mesh untuk mendeteksi 468+ titik landmark wajah.
- **Scale Invariant Feature Extraction**: Menghitung metrik rasio geometris wajah (dahi, pipi, rahang, dan tinggi wajah) yang ternormalisasi sehingga deteksi tetap akurat meskipun jarak wajah ke kamera berubah-ubah.
- **Random Forest Classifier**: Mengklasifikasikan bentuk wajah ke dalam 6 tipe (Oval, Round, Square, Heart, Diamond, Oblong) dengan tingkat akurasi mencapai **81.97%**!
- **Rule-Based Fallback**: Jika model Machine Learning belum dilatih, sistem akan mendeteksi bentuk wajah menggunakan hitungan rasio geometric rule-based sebagai cadangan otomatis.
- **Unified Interactive CLI Menu**: Menu utama yang interaktif memudahkan Anda mengelola seluruh pipeline aplikasi tanpa perlu mengetikkan command yang panjang di terminal.

---

## Struktur Folder Proyek

```text
Sistem Rekomendasi Model Cukur Rambut/
│
├── Data/                     # Folder utama dataset gambar wajah mentah
│   ├── Oval/                 # Foto berwajah Oval
│   ├── Round/                # Foto berwajah Bulat
│   ├── Square/               # Foto berwajah Kotak
│   ├── Heart/                # Foto berwajah Hati
│   ├── Diamond/              # Foto berwajah Wajik
│   ├── Oblong/               # Foto berwajah Panjang/Lonjong
│   ├── Mentahan/             # Foto baru yang belum dikelompokkan
│   └── Unclassified/         # Foto yang gagal mendeteksi wajah/titik landmark
│
├── dataset/                  # File fitur hasil ekstraksi
│   └── labels.csv            # File CSV berisi metrik wajah yang diekstrak
│
├── models/
│   ├── random_forest.pkl     # Model Random Forest Classifier yang terlatih
│   ├── scaler.pkl            # StandardScaler untuk normalisasi metrik wajah
│   └── feature_importance.png # Grafik analisis pentingnya fitur wajah
│
├── src/                      # Source code aplikasi utama
│   ├── camera.py             # Modul capture webcam (HD Widescreen)
│   ├── face_mesh.py          # Pemrosesan MediaPipe Face Mesh
│   ├── feature_extractor.py  # Ekstraksi fitur geometris wajah (scale-invariant)
│   ├── classifier.py         # Prediksi bentuk wajah (RF + rule fallback)
│   ├── recommender.py        # Aturan rekomendasi gaya rambut
│   ├── ui.py                 # Rendering overlay HUD visual & FPS
│   └── main.py               # Program utama webcam rekomendasi
│
├── training/                 # Pipeline pelatihan & utilitas dataset
│   ├── classify_and_sort.py  # Menyortir foto otomatis dari Mentahan/ ke folder kelas
│   ├── augment_data.py       # Augmentasi data gambar (mirror, brightness, rotasi)
│   ├── prepare_dataset.py    # Ekstraksi fitur gambar menjadi data tabular di CSV
│   ├── train_model.py        # Melatih model Random Forest
│   └── evaluate.py           # Evaluasi metrik & akurasi klasifikasi
│
├── tests/
│   └── test_components.py    # Unit testing untuk modul backend
│
├── main.py                   # Menu Kontrol Utama (Interactive CLI Menu)
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Petunjuk penggunaan proyek
```

---

## Persiapan Instalasi

### Prasyarat
- Python 3.11 atau versi di atasnya
- Webcam yang berfungsi

### Langkah Instalasi
1. Buka terminal pada direktori proyek.
2. Instal semua dependensi library yang dibutuhkan menggunakan pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## Petunjuk Penggunaan (Menu Utama)

Untuk memulai, cukup jalankan menu kontrol utama dari root direktori proyek:
```bash
python main.py
```

Anda akan disajikan dengan menu interaktif berikut:
```text
=============================================================
============== HairStyle Recommendation System ==============
=============================================================
 1. Start HairStyle Recommendation
 2. Pelatihan Model (Auto-sort -> Ekstrak -> Train -> Eval)
 3. Lihat Akurasi AI Bentuk Wajah
 4. Keluar
=============================================================
Masukkan pilihan (1-4):
```

### Penjelasan Menu:

#### Opsi 1: Start HairStyle Recommendation
Menjalankan kamera real-time secara langsung untuk mendeteksi wajah dan memberikan rekomendasi gaya rambut.
- **`[F]`**: Tombol pintas untuk beralih mode Fullscreen / Windowed.
- **`[M]`**: Tombol pintas untuk mengaktifkan/menonaktifkan garis pandu pengukuran bentuk wajah.
- **`[ESC]`**: Tombol pintas untuk keluar dari kamera dan kembali ke menu utama.

#### Opsi 2: Pelatihan Model
Pipeline pelatihan lengkap yang berjalan secara otomatis berurutan:
1. Menyortir foto di folder `Data/Mentahan/` ke masing-masing kelas wajah secara cerdas.
2. Mengekstrak metrik wajah ke `dataset/labels.csv`.
3. Melatih ulang model Machine Learning Random Forest.
4. Menampilkan statistik performa evaluasi yang baru.

#### Opsi 3: Lihat Akurasi AI Bentuk Wajah
Membaca model yang ada dan langsung menampilkan laporan metrik evaluasi model (Precision, Recall, F-Score, Confusion Matrix) secara instan tanpa perlu pelatihan ulang.

#### Opsi 4: Keluar
Menutup program menu interaktif.

---

## Unit Testing

Untuk memastikan semua modul berjalan dengan benar, jalankan unit test berikut:
```bash
python -m unittest tests/test_components.py
```
