"""Evaluasi hasil OCR: pola format plat Indonesia + confidence (rule-based).

Format: 1-2 huruf (wilayah) + 1-4 angka + 0-3 huruf (seri). Contoh: P 4602 IN.
Karakter yang tertukar dikoreksi berdasarkan POSISI (huruf di blok angka -> angka, dan sebaliknya)."""
import re

from app.config import PLATE_LENGTH, PLATE_MIN_CONFIDENCE

_KE_HURUF = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
_KE_ANGKA = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


def normalisasi(teks):
    return re.sub(r"[^A-Z0-9]", "", (teks or "").upper())


def koreksi_posisi(teks):
    """Cari pembagian huruf-angka-huruf dengan koreksi paling sedikit.
    Return (teks_terkoreksi, jumlah_koreksi) atau (None, None) bila tak bisa sesuai format."""
    n, terbaik = len(teks), None
    if not PLATE_LENGTH[0] <= n <= PLATE_LENGTH[1]:
        return None, None
    for lp in (1, 2):
        for ld in range(1, 5):
            ls = n - lp - ld
            if not 0 <= ls <= 3:
                continue
            out, biaya, ok = [], 0, True
            for i, ch in enumerate(teks):
                blok_huruf = i < lp or i >= lp + ld
                if blok_huruf and ch.isdigit():
                    if ch not in _KE_HURUF:
                        ok = False
                        break
                    ch, biaya = _KE_HURUF[ch], biaya + 1
                elif not blok_huruf and ch.isalpha():
                    if ch not in _KE_ANGKA:
                        ok = False
                        break
                    ch, biaya = _KE_ANGKA[ch], biaya + 1
                out.append(ch)
            if ok and (terbaik is None or biaya < terbaik[1]):
                terbaik = ("".join(out), biaya)
    return terbaik if terbaik else (None, None)


def evaluasi(hasil_ocr):
    """Beri status tiap hasil OCR lalu urutkan (terbaik dulu).
    valid = sesuai format DAN confidence >= PLATE_MIN_CONFIDENCE."""
    unik = {}
    for r in hasil_ocr:
        mentah = normalisasi(r["text"])
        if not mentah:
            continue
        teks, koreksi = koreksi_posisi(mentah)
        format_ok = teks is not None
        item = {
            "text": teks if format_ok else mentah,
            "raw": mentah,
            "confidence": round(float(r["confidence"]), 3),
            "corrections": koreksi if format_ok else None,
            "format_ok": format_ok,
            "valid": format_ok and r["confidence"] >= PLATE_MIN_CONFIDENCE,
            "variant": r.get("variant"),
            "box": r.get("box"),
        }
        lama = unik.get(item["text"])
        if lama is None or item["confidence"] > lama["confidence"]:
            unik[item["text"]] = item
    return sorted(unik.values(), key=lambda x: (not x["valid"], x["corrections"] if x["corrections"] is not None else 99, -x["confidence"]))
