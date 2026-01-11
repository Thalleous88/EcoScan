# EcoScan: Multi-Modal AI Diagnostic Tool

**EcoScan** adalah platform diagnostik berbasis kecerdasan buatan (AI) yang mengintegrasikan berbagai model *Deep Learning* untuk mendeteksi jenis perangkat, mengklasifikasi cacat visual, dan memberikan solusi perbaikan secara otomatis. Proyek ini menggabungkan teknik *Computer Vision* dan *Natural Language Processing* (NLP) dalam satu alur kerja yang efisien.

---

## 🚀 Fitur Utama

- **Deteksi Perangkat (YOLOv8):** Mendeteksi jenis perangkat keras secara akurat.
- **Klasifikasi Cacat (ResNet50):** Menganalisis kerusakan fisik atau cacat pada permukaan perangkat.
- **Analisis Keluhan (DistilBERT):** Memahami konteks keluhan tekstual dari pengguna.
- **Sistem Rekomendasi (Qwen 2.5 1.5B):** Memberikan saran perbaikan teknis berbasis Large Language Model (LLM).

---

## 🛠️ Panduan Instalasi

Pastikan Anda sudah menginstal **Python 3.10+** dan **Git** di sistem Anda. Ikuti langkah-langkah berikut:

### 1. Clone Repositori

Gunakan terminal untuk mengunduh repositori ini ke komputer Anda:
```bash
git clone https://github.com/Thalleous88/ecoscan.git
cd ecoscan
```

### 2. Membuat Virtual Environment (Disarankan)

Agar library tidak bentrok dengan proyek lain:
```bash
# Untuk Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# Untuk Windows:
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalasi Dependensi

Instal semua library yang diperlukan menggunakan berkas requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Cara Menjalankan Aplikasi

Setelah instalasi selesai, Anda dapat menjalankan antarmuka web EcoScan yang berbasis Streamlit dengan perintah:
```bash
streamlit run app2.py
```

Setelah itu, buka browser Anda dan akses: **http://localhost:8501**

---

## 📂 Struktur Folder
```
ecoscan/
├── app2.py                         # File utama aplikasi Streamlit
├── requirements.txt                # Daftar library Python
├── electronics_type_classifier/    # Model object detection (YOLOv8)
├── condition_classifier/           # Model image classification (ResNet50)
├── keyword_extraction/             # Model NLP (DistilBERT)
├── backend2.py/                    # integrasi semua model dengan LLM (Qwen 2.5 1.5B Instruct) dan front-end streamlit
└── README.md                       # Dokumentasi proyek
```
