"""OCR plat: EasyOCR atau Tesseract (dipilih lewat config PLATE_OCR_ENGINE)."""
import re
from difflib import SequenceMatcher

from app.config import PLATE_ALLOWLIST, PLATE_OCR_ENGINE, PLATE_TESSERACT_CMD

_engine = None
_easy_reader = None


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


def _baca_easyocr(img):
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        _easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    hasil = _easy_reader.readtext(img, allowlist=PLATE_ALLOWLIST, detail=1, paragraph=False)
    if not hasil:
        return "", 0.0
    hasil.sort(key=lambda r: min(p[0] for p in r[0]))   # kiri ke kanan
    n = sum(len(r[1]) for r in hasil) or 1
    return "".join(r[1] for r in hasil), sum(r[2] * len(r[1]) for r in hasil) / n


def _data_tesseract(img, cfg):
    import pytesseract
    d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    return [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]


def _rata_conf(bagian):
    n = sum(len(t) for t, _ in bagian)
    return sum(c * len(t) for t, c in bagian) / n / 100.0 if n else 0.0


def _baca_tesseract(img):
    """Dua kali baca: (a) dengan whitelist karakter -> teks lebih bersih;
    (b) tanpa whitelist -> confidence. Tesseract 5 melaporkan confidence 0 saat whitelist aktif,
    jadi confidence akhir = confidence (b) x kemiripan teks (a) dan (b)."""
    import pytesseract
    if PLATE_TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = PLATE_TESSERACT_CMD
    dasar = "--oem 3 --psm 7"
    dengan = _data_tesseract(img, f"{dasar} -c tessedit_char_whitelist={PLATE_ALLOWLIST}")
    tanpa = _data_tesseract(img, dasar)

    teks_w = "".join(t for t, _ in dengan)
    teks_n = re.sub(r"[^A-Z0-9]", "", "".join(t for t, _ in tanpa).upper())
    if dengan and max(c for _, c in dengan) > 0:          # build Tesseract yang melaporkan confidence
        return teks_w, _rata_conf(dengan)
    teks = teks_w or teks_n
    if not teks:
        return "", 0.0
    kemiripan = SequenceMatcher(None, teks, teks_n).ratio() if teks_n else 0.0
    return teks, _rata_conf(tanpa) * kemiripan


def read_variants(varian):
    """Return list dict(text, confidence, variant, engine) dari semua varian citra."""
    engine = _pilih_engine()
    baca = _baca_easyocr if engine == "easyocr" else _baca_tesseract
    hasil = []
    for nama, img in varian:
        teks, conf = baca(img)
        if teks:
            hasil.append({"text": teks, "confidence": float(conf), "variant": nama, "engine": engine})
    return hasil
