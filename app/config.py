from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "models" / "lbph.yml"

FACE_SIZE = (200, 200)
MIN_FACE_PX = 80
LBPH_PARAMS = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)
UNKNOWN_THRESHOLD = 70.0   # makin kecil makin mirip; kalibrasi dengan data nyata

# Pendaftaran: browser mengirim ~30 frame, tidak semuanya berisi wajah.
MIN_VALID_FRAMES = 30      # minimal frame yang wajahnya terdeteksi Haar cascade
# Dengan 30 frame asli, augmentasi tidak wajib. Tiap citra training memakan
# ~64 KB di model (grid 8x8), jadi augmentasi mempercepat membengkaknya model.
AUGMENT = False

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
PLATE_CLOSE_KERNEL = (17, 3)            # morphological closing (lebar > tinggi: menyambung tepi karakter)
PLATE_ASPECT_RANGE = (2.0, 5.5)         # lebar:tinggi (plat mobil ~3,2; motor ~2,4)
PLATE_AREA_RATIO = (0.004, 0.30)        # luas kandidat relatif luas citra
PLATE_MIN_RECTANGULARITY = 0.70         # luas kontur / luas persegi panjang terkecil
PLATE_MAX_CANDIDATES = 3                # kandidat lokasi yang dicoba ke OCR
PLATE_CROP_MARGIN = 0.04                # tambahan sisi saat crop

# 2) Preprocessing lanjutan sebelum OCR
PLATE_OCR_WIDTH = 400                   # lebar plat setelah resize

# 3) OCR
PLATE_OCR_ENGINE = "auto"               # "auto" | "easyocr" | "tesseract"
PLATE_TESSERACT_CMD = ""                # Windows, mis. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_MIN_CONFIDENCE = 0.40             # di bawah ini dianggap tidak yakin
PLATE_LENGTH = (4, 9)                   # jumlah karakter (tanpa spasi) yang masuk akal: "B1AB" ... "AB1234CDE"
