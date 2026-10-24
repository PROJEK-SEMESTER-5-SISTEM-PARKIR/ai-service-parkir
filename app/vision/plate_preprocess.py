"""Preprocessing lanjutan citra plat (hasil crop) sebelum OCR.

Alur: resize -> grayscale -> pengurangan noise -> peningkatan kontras (CLAHE)
      -> thresholding. Dihasilkan beberapa varian; OCR mencoba semuanya dan
      hasil dengan confidence terbaik yang dipakai."""
import cv2

from app.config import PLATE_OCR_WIDTH

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _teks_gelap_latar_terang(biner):
    """Plat Indonesia umumnya dasar hitam, teks putih. Setelah threshold, dasar
    mendominasi -> bila gambar mayoritas gelap, dibalik supaya teks gelap di latar terang."""
    return cv2.bitwise_not(biner) if biner.mean() < 127 else biner


def _pad(img):
    return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def prepare_variants(plate_bgr, debug=None):
    """Return list (nama, citra_grayscale/biner) untuk OCR."""
    h, w = plate_bgr.shape[:2]
    s = PLATE_OCR_WIDTH / float(w)
    resized = cv2.resize(plate_bgr, (PLATE_OCR_WIDTH, max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 60, 60)   # kurangi noise, tepi karakter tetap tajam
    kontras = _clahe.apply(denoised)

    _, otsu = cv2.threshold(kontras, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptif = cv2.adaptiveThreshold(kontras, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    otsu = _teks_gelap_latar_terang(otsu)
    adaptif = _teks_gelap_latar_terang(adaptif)

    kontras_ocr = cv2.bitwise_not(kontras) if kontras.mean() < 127 else kontras
    varian = [("clahe", _pad(kontras_ocr)), ("otsu", _pad(otsu)), ("adaptif", _pad(adaptif))]
    if debug is not None:
        debug.update({"08_resize_gray": gray, "09_denoise": denoised, "10_clahe": kontras,
                      "11_threshold_otsu": otsu, "12_threshold_adaptif": adaptif})
    return varian
