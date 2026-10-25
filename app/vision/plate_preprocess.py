"""Preprocessing citra plat (hasil crop) sebelum OCR.

Tujuannya MEMBANTU OCR, bukan sekadar memproses:
  resize -> grayscale -> bilateral (kurangi noise) -> CLAHE (ratakan kontras)
  -> segmentasi karakter -> gambar ulang karakter di kanvas putih bersih.
Bingkai, baut, dan baris masa berlaku ("09-23") otomatis hilang karena bukan karakter.
Bila segmentasi gagal, dipakai varian cadangan (CLAHE / Otsu / adaptif).

Varian tambahan:
  - sharp        : unsharp masking untuk foto blur/low-res
  - gamma_bright : koreksi gamma (γ=0.5) untuk plat gelap/kondisi malam
  - morph_close  : closing setelah threshold agar karakter putus tersambung
  - bersih_150, bersih_200 : multi-scale 1.5x/2x (Tesseract lebih akurat di teks lebih besar)
"""
import cv2
import numpy as np

from app.config import (PLATE_OCR_SCALES, PLATE_OCR_WIDTH, PLATE_SHARPENING,
                        PLATE_TINGGI_KARAKTER)
from app.vision import plate_segment

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_clahe_kuat = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  # untuk foto sangat gelap


def _pad(img, n=15):
    return cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_CONSTANT, value=255)


def _skala(img, faktor):
    interp = cv2.INTER_CUBIC if faktor > 1 else cv2.INTER_AREA
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * faktor)), max(1, int(h * faktor))), interpolation=interp)


def _teks_gelap_latar_terang(biner):
    return cv2.bitwise_not(biner) if biner.mean() < 127 else biner


def _unsharp_mask(gray, sigma=1.0, kuat=1.5):
    """Unsharp masking: mempertajam tepi karakter yang blur."""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + kuat, blur, -kuat, 0)


def _gamma(gray, gamma=0.5):
    """Koreksi gamma: gamma < 1 mencerahkan, gamma > 1 menggelapkan."""
    lut = np.array([min(255, int(pow(i / 255.0, gamma) * 255)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut)


def _morph_close(biner, kernel_size=3):
    """Morphological closing: menyambung karakter yang sedikit terputus."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    return cv2.morphologyEx(biner, cv2.MORPH_CLOSE, kernel)


def _varian_cadangan(kontras, debug):
    _, otsu = cv2.threshold(kontras, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptif = cv2.adaptiveThreshold(kontras, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    otsu = _teks_gelap_latar_terang(otsu)
    adaptif = _teks_gelap_latar_terang(adaptif)
    kontras_ocr = cv2.bitwise_not(kontras) if kontras.mean() < 127 else kontras

    # Varian tambahan: sharp dan gamma bright
    sharp = _unsharp_mask(kontras_ocr) if PLATE_SHARPENING else kontras_ocr
    gamma_bright = _gamma(kontras_ocr, gamma=0.5)
    otsu_morph = _morph_close(_teks_gelap_latar_terang(
        cv2.threshold(kontras, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))

    if debug is not None:
        debug.update({
            "11_threshold_otsu": otsu,
            "12_threshold_adaptif": adaptif,
            "13_sharp": sharp,
            "14_gamma_bright": gamma_bright,
        })
    return [
        ("clahe",       _pad(kontras_ocr, 10)),
        ("otsu",        _pad(otsu, 10)),
        ("adaptif",     _pad(adaptif, 10)),
        ("sharp",       _pad(sharp, 10)),
        ("gamma_bright",_pad(gamma_bright, 10)),
        ("otsu_morph",  _pad(otsu_morph, 10)),
    ]


def _varian_segmentasi(seg, debug):
    grup, labels, gray = seg["grup"], seg["labels"], seg["gray"]
    Hh, Ww = gray.shape
    x_min = min(k[1] for k in grup)
    y_min = min(k[2] for k in grup)
    x_max = max(k[1] + k[3] for k in grup)
    y_max = max(k[2] + k[4] for k in grup)
    med_h = float(np.median([k[4] for k in grup]))
    faktor = PLATE_TINGGI_KARAKTER / med_h

    def kotak(m):
        mm = int(m * med_h)
        return max(0, x_min - mm), max(0, y_min - mm), min(Ww, x_max + mm), min(Hh, y_max + mm)

    # 1) karakter digambar ulang hitam di kanvas putih
    kanvas = np.full((Hh, Ww), 255, dtype=np.uint8)
    for k in grup:
        kanvas[labels == k[0]] = 0
    x0, y0, x1, y1 = kotak(0.35)
    bersih_base = _skala(kanvas[y0:y1, x0:x1], faktor)
    bersih = _pad(bersih_base)
    halus = cv2.GaussianBlur(bersih, (3, 3), 0)   # tepi lebih lembut, goresan tipis ('1') tidak putus

    # 2) grayscale asli (CLAHE) di area karakter, teks dibuat gelap di latar terang
    x0, y0, x1, y1 = kotak(0.2)
    abu = gray[y0:y1, x0:x1]
    if seg["nama"].endswith("+"):                  # karakter putih -> balik
        abu = cv2.bitwise_not(abu)
    abu_base = _skala(abu, faktor)
    abu = _pad(abu_base)

    # 3) Varian tambahan: unsharp masking pada abu (memperjelas tepi karakter blur)
    abu_sharp_src = cv2.bitwise_not(abu_base) if seg["nama"].endswith("+") else abu_base
    sharp_abu = _pad(_unsharp_mask(abu_sharp_src, sigma=0.8, kuat=1.2)) if PLATE_SHARPENING else abu

    # 4) Gamma correction pada abu (membantu kondisi gelap)
    gamma_img = _pad(_gamma(abu_base, gamma=0.6))

    # 5) Morphological close pada bersih_base (menyambung karakter putus)
    morph_bersih = _pad(_morph_close(bersih_base))

    # 6) Multi-scale: versi 1.5x dan 2x dari bersih (Tesseract lebih akurat pada teks lebih besar)
    varian_scale = []
    for skala_f in PLATE_OCR_SCALES:
        if skala_f == 1.0:
            continue
        nama_scale = f"bersih_{int(skala_f * 100)}"
        varian_scale.append((nama_scale, _pad(_skala(bersih_base, skala_f))))

    if debug is not None:
        debug.update({
            "11_biner_terpilih": seg["biner"],
            "12_karakter_bersih": kanvas[y0:y1, x0:x1] if False else bersih_base,
            "13_abu_crop": abu_base,
            "14_sharp_abu": sharp_abu,
            "15_gamma_bright": gamma_img,
        })

    hasil = [
        ("bersih",      bersih),
        ("halus",       halus),
        ("abu",         abu),
        ("sharp",       sharp_abu),
        ("gamma",       gamma_img),
        ("morph_bersih",morph_bersih),
    ]
    hasil.extend(varian_scale)
    return hasil


def prepare_variants(plate_bgr, debug=None, fast=False):
    """Return list (nama, citra) untuk OCR.

    fast=True : hanya 3 varian utama (bersih_150, abu, sharp) - untuk real-time kamera.
    fast=False: semua varian (8+) - untuk akurasi tinggi pada foto statis.
    """
    h, w = plate_bgr.shape[:2]
    s = PLATE_OCR_WIDTH / float(w)
    resized = _skala(plate_bgr, s)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Bilateral filter: d=7 lebih ringan dari 9, kurangi risiko melumuri tepi karakter tipis
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    kontras = _clahe.apply(denoised)
    if debug is not None:
        debug.update({"08_resize_gray": gray, "09_denoise": denoised, "10_clahe": kontras})

    seg = plate_segment.cari_karakter(kontras)
    if seg is None:
        varian = _varian_cadangan(kontras, debug)
    else:
        varian = _varian_segmentasi(seg, debug)

    # Mode cepat: ambil 3 varian terpilih saja
    if fast:
        PRIORITAS = ("bersih_150", "bersih", "sharp", "abu", "clahe", "otsu")
        pilihan = [v for nama in PRIORITAS for v in varian if v[0] == nama]
        return pilihan[:3] if pilihan else varian[:3]
    return varian