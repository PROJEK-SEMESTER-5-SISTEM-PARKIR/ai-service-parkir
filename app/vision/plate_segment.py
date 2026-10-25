"""Segmentasi karakter plat.

Dipakai dua tempat:
  - plate_detector: menilai kandidat lokasi (berapa karakter di dalam crop?)
  - plate_preprocess: membersihkan crop (buang bingkai, baut, baris masa berlaku)
"""
import cv2
import numpy as np

TINGGI_MIN, TINGGI_MAX = 0.22, 0.80   # tinggi karakter relatif tinggi crop
ASPEK_MIN, ASPEK_MAX = 0.08, 1.2      # lebar/tinggi karakter ('1' kurus, 'W' lebar)
MAKS_KARAKTER = 10
JARAK_MAKS = 1.8                      # diturunkan: batas max jarak antar karakter agar baris ke-2 tidak masuk


def kelompok_baris(arr):
    """arr: array Nx4 (x, y, w, h). Cari kelompok kotak yang tingginya mirip, sejajar,
    dan berderet rapat (satu baris karakter). Return list kelompok (list indeks, urut x).
    Filter tambahan: karakter yang y-center-nya berbeda > 40% tinggi median grup dibuang."""
    hasil, terlihat = [], set()
    if len(arr) == 0:
        return hasil
    cy = arr[:, 1] + arr[:, 3] / 2
    for a in range(len(arr)):
        ha = arr[a, 3]
        cocok = (arr[:, 3] >= 0.75 * ha) & (arr[:, 3] <= 1.33 * ha) & (np.abs(cy - cy[a]) <= 0.3 * ha)
        idx = np.where(cocok)[0]
        idx = idx[np.argsort(arr[idx, 0])]
        run, runs = [int(idx[0])], []
        for j in idx[1:]:
            tepi_kanan = arr[run[-1], 0] + arr[run[-1], 2]
            if arr[j, 0] - tepi_kanan > JARAK_MAKS * ha:
                runs.append(run)
                run = [int(j)]
            else:
                run.append(int(j))
        runs.append(run)
        for r in runs:
            # Filter y-center: buang karakter yang y-center-nya menyimpang > 40% tinggi median
            if len(r) >= 3:
                cy_run = np.array([arr[i, 1] + arr[i, 3] / 2 for i in r])
                h_med = float(np.median([arr[i, 3] for i in r]))
                cy_med = float(np.median(cy_run))
                r = [i for i, cyv in zip(r, cy_run) if abs(cyv - cy_med) <= 0.4 * h_med]
            kunci = tuple(r)
            if kunci not in terlihat:
                terlihat.add(kunci)
                hasil.append(r)
    return hasil


def _biner_varian(gray):
    """Empat biner. '+' = karakter putih (plat gelap), '-' = karakter gelap (plat terang)."""
    h, w = gray.shape
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blok = max(15, (min(h, w) // 4) | 1)
    adaptif = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blok, 8)
    return [("otsu+", otsu), ("otsu-", cv2.bitwise_not(otsu)),
            ("adaptif+", adaptif), ("adaptif-", cv2.bitwise_not(adaptif))]


def _komponen(biner):
    H, W = biner.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    hasil = []
    for i in range(1, n):
        x, y, w, h, luas = (int(v) for v in stats[i])
        if not (TINGGI_MIN * H <= h <= TINGGI_MAX * H):
            continue
        if not (ASPEK_MIN <= w / h <= ASPEK_MAX):
            continue
        if luas < 0.2 * w * h:
            continue
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:   # menempel tepi = bingkai / terpotong
            continue
        hasil.append((i, x, y, w, h))
    return labels, hasil


def cari_karakter(gray):
    """Cari satu baris karakter pada crop plat (grayscale).
    Return dict(n, grup, labels, biner, nama, gray) atau None bila tidak ada baris karakter.
    `grup` = list (label, x, y, w, h) dalam koordinat citra yang sudah diberi padding (`gray`)."""
    h0, w0 = gray.shape
    if h0 < 12 or w0 < 24:
        return None
    pad = max(4, int(0.12 * h0))
    ring = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.median(ring)))
    H_g = g.shape[0]

    # Batas y-center: masa berlaku selalu di bawah ~68-75% tinggi plat.
    # Karakter dengan y-center di atas batas ini langsung dibuang.
    Y_CENTER_MAKS = 0.72 * H_g

    terbaik = None
    for nama, biner in _biner_varian(g):
        labels, kand = _komponen(biner)
        if not kand:
            continue

        # Buang karakter yang y-center-nya di bawah batas (masa berlaku)
        kand = [k for k in kand if (k[2] + k[4] / 2) <= Y_CENTER_MAKS]
        if not kand:
            continue

        arr = np.array([[k[1], k[2], k[3], k[4]] for k in kand], dtype=float)
        runs = kelompok_baris(arr)
        if not runs:
            continue
        run = max(runs, key=len)
        n = len(run)
        if not (3 <= n <= MAKS_KARAKTER):
            continue
        grup = [kand[j] for j in run]
        tinggi = np.array([k[4] for k in grup], dtype=float)

        # Penalti skor bila grup masih ada di bawah 60% tinggi crop
        cy_grup = float(np.mean([k[2] + k[4] / 2 for k in grup]))
        penalti = 0.5 if cy_grup > 0.60 * H_g else 1.0

        skor = n * (1.0 - min(0.5, float(tinggi.std() / tinggi.mean()))) * penalti
        if terbaik is None or skor > terbaik["skor"]:
            terbaik = {"n": n, "grup": grup, "labels": labels, "biner": biner,
                       "nama": nama, "gray": g, "skor": skor}
    return terbaik