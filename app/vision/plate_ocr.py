"""OCR plat: EasyOCR atau Tesseract (dipilih lewat config PLATE_OCR_ENGINE).

Perbaikan Tesseract:
  - Mode PSM yang dicoba: 7 (satu baris), 8 (satu kata), 6 (blok seragam), 13 (raw line)
  - OEM 3 (LSTM + Legacy) untuk akurasi terbaik
  - Whitelist karakter aktif; fallback tanpa whitelist bila confidence 0
  - Hanya jalankan scale yang ada di PLATE_OCR_SCALES bila gambar sudah di-scale

Setiap varian gambar bisa menghasilkan beberapa bacaan (kombinasi PSM × scale).
plate_rules.evaluasi() kemudian memilih yang terbaik via voting per karakter.
"""
import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from app.config import (PLATE_ALLOWLIST, PLATE_MIN_CONFIDENCE, PLATE_OCR_ENGINE,
                        PLATE_TESSERACT_CMD, PLATE_TESSERACT_PSMS, PLATE_TESSDATA_DIR)

_engine = None
_easy_reader = None

# Teks dengan tinggi < 70% tinggi teks terbesar dianggap bukan nomor plat.
RASIO_TINGGI_MIN = 0.7


def _pilih_engine():
    global _engine
    if _engine:
        return _engine
    if PLATE_OCR_ENGINE in ("easyocr", "tesseract"):
        _engine = PLATE_OCR_ENGINE
        return _engine
    try:
        import easyocr  # noqa: F401
        _engine = "easyocr"
    except ImportError:
        try:
            import pytesseract  # noqa: F401
            _engine = "tesseract"
        except ImportError:
            raise RuntimeError("OCR belum terpasang. Pasang pytesseract (+ program Tesseract) atau easyocr.")
    return _engine


def _baca_easyocr(img, psm=None):
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        _easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    # Parameter yang dioptimalkan untuk teks plat kecil
    hasil = _easy_reader.readtext(
        img,
        allowlist=PLATE_ALLOWLIST,
        detail=1,
        paragraph=False,
        text_threshold=0.6,    # minimum confidence teks
        low_text=0.3,          # sensitivitas teks kecil
        link_threshold=0.4,    # penghubung antar karakter
        width_ths=0.7,         # agar kata berdekatan tidak digabung salah
    )
    if not hasil:
        return "", 0.0

    def tinggi(r):
        ys = [p[1] for p in r[0]]
        return max(ys) - min(ys)

    tmax = max(tinggi(r) for r in hasil)
    hasil = [r for r in hasil if tinggi(r) >= RASIO_TINGGI_MIN * tmax]
    hasil.sort(key=lambda r: min(p[0] for p in r[0]))   # kiri ke kanan
    n = sum(len(r[1]) for r in hasil) or 1
    return "".join(r[1] for r in hasil), sum(r[2] * len(r[1]) for r in hasil) / n


def _atur_tesseract():
    import pytesseract
    if PLATE_TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = PLATE_TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = PLATE_TESSDATA_DIR


def _data_tesseract(img, cfg):
    import pytesseract
    d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    kata = [(t.strip(), float(c), int(h))
            for t, c, h in zip(d["text"], d["conf"], d["height"])
            if t.strip() and float(c) >= 0]
    if not kata:
        return []
    hmax = max(h for _, _, h in kata)
    return [(t, c) for t, c, h in kata if h >= RASIO_TINGGI_MIN * hmax]


def _rata_conf(bagian):
    n = sum(len(t) for t, _ in bagian)
    return sum(c * len(t) for t, c in bagian) / n / 100.0 if n else 0.0


def _baca_tesseract(img, psm=7):
    """Baca plat dengan whitelist karakter.

    Strategi:
    1. Coba whitelist; bila confidence > 0 → kembalikan langsung.
    2. Bila confidence = 0 (bug Tesseract 5 dengan whitelist) → coba tanpa whitelist,
       lalu timbang confidence dengan kemiripan teks.
    3. Bila hasil masih kosong → kembalikan ('', 0.0).
    """
    _atur_tesseract()
    # OEM 3 = LSTM + Legacy (akurasi terbaik), PSM dari parameter
    dasar = f"--oem 3 --psm {psm}"
    cfg_whitelist = f"{dasar} -c tessedit_char_whitelist={PLATE_ALLOWLIST}"

    dengan = _data_tesseract(img, cfg_whitelist)
    if dengan and max(c for _, c in dengan) > 0:
        teks = "".join(t for t, _ in dengan)
        conf = _rata_conf(dengan)
        return teks, conf

    # Fallback: tanpa whitelist
    tanpa = _data_tesseract(img, dasar)
    teks_w = re.sub(r"[^A-Z0-9]", "", "".join(t for t, _ in dengan).upper())
    teks_n = re.sub(r"[^A-Z0-9]", "", "".join(t for t, _ in tanpa).upper())
    teks = teks_w or teks_n
    if not teks:
        return "", 0.0
    kemiripan = SequenceMatcher(None, teks, teks_n).ratio() if teks_n else 0.5
    conf_tanpa = _rata_conf(tanpa) if tanpa else 0.0
    return teks, conf_tanpa * kemiripan


def read_variants(varian, fast=False):
    """Return list dict(text, confidence, variant, engine) dari semua varian citra.

    fast=True : hanya PSM 7 (1 call per varian) - untuk real-time kamera.
    fast=False: semua PSM di PLATE_TESSERACT_PSMS (4 call per varian) - untuk akurasi tinggi.
    """
    engine = _pilih_engine()

    if engine == "tesseract":
        _atur_tesseract()
        mode = (7,) if fast else PLATE_TESSERACT_PSMS   # fast: PSM 7 saja
        baca = _baca_tesseract
    else:
        mode = (None,)
        baca = _baca_easyocr

    hasil = []
    for nama, img in varian:
        if engine == "tesseract" and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for psm in mode:
            teks, conf = baca(img, psm)
            teks = re.sub(r"[^A-Z0-9]", "", teks.upper()) if teks else ""
            if not teks:
                continue
            label = f"{nama}/psm{psm}" if psm is not None else nama
            hasil.append({
                "text": teks,
                "confidence": float(conf),
                "variant": label,
                "engine": engine,
            })

    return hasil