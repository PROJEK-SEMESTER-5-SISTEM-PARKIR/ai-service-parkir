"""Evaluasi hasil OCR: pola format plat Indonesia + confidence (rule-based).

Format: 1-2 huruf (wilayah) + 1-4 angka + 0-3 huruf (seri). Contoh: P 4602 IN.
Karakter yang tertukar dikoreksi berdasarkan POSISI (huruf di blok angka -> angka, dan sebaliknya).
Bacaan dari banyak varian digabung dengan voting per karakter."""
import re
from collections import defaultdict

import numpy as np

from app.config import PLATE_LENGTH, PLATE_MIN_CONFIDENCE


# Koreksi karakter yang sering tertukar oleh OCR pada plat nomor Indonesia.
# _KE_HURUF : angka di posisi yang seharusnya huruf -> huruf yang mirip
# _KE_ANGKA : huruf di posisi yang seharusnya angka -> angka yang mirip
_KE_HURUF = {
    "0": "O",   # 0 -> O
    "1": "I",   # 1 -> I
    "2": "Z",   # 2 -> Z
    "3": "B",   # 3 -> B (plat Jawa Tengah: huruf awal bisa B)
    "4": "A",   # 4 -> A
    "5": "S",   # 5 -> S
    "6": "G",   # 6 -> G
    "7": "T",   # 7 -> T (sangat sering di font plat)
    "8": "B",   # 8 -> B
    "9": "P",   # 9 -> P (mirip bentuk)
}
_KE_ANGKA = {
    "O": "0",   # O -> 0
    "Q": "0",   # Q -> 0
    "D": "0",   # D -> 0
    "I": "1",   # I -> 1
    "L": "1",   # L -> 1
    "Z": "2",   # Z -> 2
    "E": "3",   # E -> 3 (font plat: E bisa jadi 3)
    "A": "4",   # A -> 4
    "S": "5",   # S -> 5
    "G": "6",   # G -> 6
    "T": "7",   # T -> 7
    "B": "8",   # B -> 8
    "P": "9",   # P -> 9
    "F": "7",   # F -> 7 (mirip pada resolusi rendah)
    "U": "0",   # U -> 0 (sangat blur)
    "C": "0",   # C -> 0
}


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
            if not 1 <= ls <= 3:   # wajib ada suffix huruf; tolak masa berlaku (OS26, dll.)
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


def kunci_urut(item):
    """Valid dulu, lalu koreksi paling sedikit, lalu confidence tertinggi."""
    koreksi = item["corrections"] if item["corrections"] is not None else 99
    return (not item["valid"], koreksi, -item["confidence"])


def _voting(bacaan):
    """bacaan: list (teks_terkoreksi, confidence), semua sesuai format.
    Ambil kelompok panjang teks dengan total bobot terbesar, lalu pilih karakter
    terbanyak (dibobot confidence) di tiap posisi. Return (teks, confidence) atau None.

    Fallback lintas panjang: bila kelompok panjang N dan N±1 sama-sama ada,
    coba gabungkan dengan membuang karakter tepi teks yang lebih panjang.
    """
    kelompok = defaultdict(list)
    for teks, conf in bacaan:
        kelompok[len(teks)].append((teks, max(conf, 0.01)))
    if not kelompok:
        return None

    # Kelompok panjang dengan bobot total terbesar
    panjang_terbaik, anggota = max(
        kelompok.items(), key=lambda kv: sum(c for _, c in kv[1])
    )

    # Fallback lintas panjang: bila total bobot kelompok dominan < 2×kelompok tetangga,
    # coba sertakan teks dari panjang ±1 dengan membuang karakter tepi (kiri atau kanan)
    for delta in (-1, 1):
        tetangga = kelompok.get(panjang_terbaik + delta, [])
        if not tetangga:
            continue
        bobot_utama = sum(c for _, c in anggota)
        bobot_tetangga = sum(c for _, c in tetangga)
        if bobot_tetangga > 0 and bobot_utama < 2 * bobot_tetangga:
            # Potong karakter paling tidak yakin (kiri atau kanan) dari teks lebih panjang
            if delta == 1:  # tetangga lebih panjang -> buang satu dari tetangga
                trim = [(t[1:] if len(t) == panjang_terbaik + 1 else t, c) for t, c in tetangga]
                trim2 = [(t[:-1] if len(t) == panjang_terbaik + 1 else t, c) for t, c in tetangga]
                anggota = anggota + trim + trim2
            else:           # tetangga lebih pendek -> buang satu dari anggota utama
                trim = [(t[1:] if len(t) == panjang_terbaik else t, c) for t, c in anggota]
                trim2 = [(t[:-1] if len(t) == panjang_terbaik else t, c) for t, c in anggota]
                anggota = tetangga + trim + trim2

    # Voting per karakter pada kelompok terpanjang yang dominan
    # Normalkan ke panjang median
    panjang_med = int(round(np.median([len(t) for t, _ in anggota])))
    anggota_norm = [(t, c) for t, c in anggota if len(t) == panjang_med]
    if not anggota_norm:
        anggota_norm = anggota

    hasil = []
    for i in range(len(anggota_norm[0][0])):
        suara = defaultdict(float)
        for teks, conf in anggota_norm:
            if i < len(teks):
                suara[teks[i]] += conf
        if suara:
            hasil.append(max(suara, key=suara.get))
    if not hasil:
        return None
    return "".join(hasil), sum(c for _, c in anggota_norm) / len(anggota_norm)


def evaluasi(hasil_ocr):
    """Beri status tiap hasil OCR lalu urutkan (terbaik dulu).
    valid = sesuai format DAN confidence >= PLATE_MIN_CONFIDENCE.
    Hasil voting (bila valid) ditaruh paling depan."""
    unik, bacaan = {}, []
    for r in hasil_ocr:
        mentah = normalisasi(r["text"])
        if not mentah:
            continue
        teks, koreksi = koreksi_posisi(mentah)
        format_ok = teks is not None
        if format_ok:
            bacaan.append((teks, float(r["confidence"])))
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

    urut = sorted(unik.values(), key=kunci_urut)

    voted = _voting(bacaan) if len(bacaan) >= 2 else None
    if voted:
        teks, conf = voted
        if conf >= PLATE_MIN_CONFIDENCE:
            voted_item = {
                "text": teks, "raw": teks, "confidence": round(conf, 3), "corrections": 0,
                "format_ok": True, "valid": True, "variant": "voting",
                "box": next((u["box"] for u in urut if u["box"]), None),
            }
            urut = [voted_item] + [u for u in urut if u["text"] != teks]
    return urut