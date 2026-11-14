from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# WAJAH (YOLO)
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = DATA_DIR / "models" / "best.pt"

YOLO_MIN_CONF = 0.25       # ambang deteksi kotak (di bawah ini diabaikan)
YOLO_ACCEPT_CONF = 0.90    # ambang diterima sebagai member. Ganti dengan hasil kalibrasi:
                           # naikkan kalau orang asing masih lolos, turunkan kalau member sering ditolak

# Nama kelas di model YOLO -> member_id di database Laravel.
# Kunci HARUS persis sama dengan nama kelas di model (cek: YOLO('models/best.pt').names)
MEMBER_MAP = {
    "Achmad-Ramadhani": 4,          # <-- GANTI angka dengan member_id asli di database
    "Dimas-Dwi-Nugroho": 3,
    "Dimas-Febrian-Firjatullah": 2,
    "Firdo-Al-Farobbi": 1,
    "Rintan-Nur-Wardani": 5,
}

# Masih dipakai face_preprocess.py / face_detector.py (jangan hapus dulu kalau file itu masih ada)
FACE_SIZE = (200, 200)
MIN_FACE_PX = 80

# ---------------------------------------------------------------------------
# PLAT NOMOR (teknik konvensional + OCR)
# ---------------------------------------------------------------------------
PLATE_TEST_DIR = DATA_DIR / "plates"    # foto uji untuk: python -m app.vision.plate_reader
DEBUG_DIR = DATA_DIR / "debug"          # gambar hasil tiap langkah

# Kamera HP (aplikasi IP Webcam): URL snapshot, dipakai endpoint /detect-phone-plate.
PHONE_CAMERA_URL = "http://192.168.1.10:8080/shot.jpg"   # GANTI sesuai IP HP-mu

# 1) Lokasi plat
PLATE_MAX_WIDTH = 1280                  # citra lebih lebar diperkecil dulu
PLATE_BLUR_KERNEL = (5, 5)              # Gaussian Blur
PLATE_CANNY = (50, 150)                 # ambang bawah, atas Canny
PLATE_CLOSE_KERNEL = (17, 3)            # closing besar: menyambung tepi karakter jadi satu blok
PLATE_CLOSE_KERNEL_KECIL = (9, 3)       # closing kecil: untuk plat yang menyatu dengan bumper
PLATE_ASPECT_RANGE = (2.0, 5.5)         # lebar:tinggi (plat mobil ~3,2; motor ~2,4)
PLATE_AREA_RATIO = (0.004, 0.95)        # luas kandidat relatif luas citra
PLATE_MIN_RECTANGULARITY = 0.70         # luas kontur / luas persegi panjang terkecil
PLATE_MAX_CANDIDATES = 5                # kandidat lokasi yang dicoba ke OCR
PLATE_CROP_MARGIN = 0.04                # tambahan sisi saat crop

# 2) Preprocessing lanjutan sebelum OCR
PLATE_OCR_WIDTH = 600                   # lebar crop plat sebelum segmentasi karakter
PLATE_TINGGI_KARAKTER = 48              # tinggi karakter (px) pada citra yang masuk OCR

# 3) OCR
PLATE_OCR_ENGINE = "tesseract"          # "auto" | "easyocr" | "tesseract"
TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")   # folder hasil instalasi Tesseract
PLATE_TESSERACT_CMD = str(TESSERACT_DIR / "tesseract.exe")
PLATE_TESSDATA_DIR = str(TESSERACT_DIR / "tessdata")
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_MIN_CONFIDENCE = 0.45             # di bawah ini dianggap tidak yakin
PLATE_EARLY_STOP = 0.75                 # berhenti mencoba kandidat lain hanya bila sudah sangat yakin
PLATE_LENGTH = (4, 9)                   # jumlah karakter (tanpa spasi) yang masuk akal

# PSM mode Tesseract yang dicoba per varian (makin banyak = makin akurat, makin lambat)
# 6 = uniform block, 7 = single line, 8 = single word, 13 = raw line
PLATE_TESSERACT_PSMS = (7, 8, 6, 13)

# Multi-scale: faktor resize gambar sebelum Tesseract (lebih besar = karakter lebih jelas)
PLATE_OCR_SCALES = (1.0, 1.5, 2.0)

# Sharpening: aktifkan unsharp masking sebelum OCR (membantu foto blur)
PLATE_SHARPENING = True