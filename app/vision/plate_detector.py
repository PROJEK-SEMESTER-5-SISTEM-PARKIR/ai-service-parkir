"""Lokasi plat nomor dengan teknik konvensional.

Alur: grayscale -> Gaussian Blur -> Canny -> morphological closing -> kontur
      -> seleksi geometri (rasio aspek, luas, kepersegian) -> crop.
"""
import cv2
import numpy as np

from app.config import (
    PLATE_ASPECT_RANGE, PLATE_AREA_RATIO, PLATE_BLUR_KERNEL, PLATE_CANNY,
    PLATE_CLOSE_KERNEL, PLATE_CROP_MARGIN, PLATE_MAX_CANDIDATES,
    PLATE_MAX_WIDTH, PLATE_MIN_RECTANGULARITY,
)


def _limit_width(img):
    h, w = img.shape[:2]
    if w <= PLATE_MAX_WIDTH:
        return img
    s = PLATE_MAX_WIDTH / w
    return cv2.resize(img, (PLATE_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    irisan = ix * iy
    return irisan / float(aw * ah + bw * bh - irisan)


def _didalam(kecil, besar, batas=0.8):
    """True bila >= `batas` luas `kecil` berada di dalam `besar`."""
    kx, ky, kw, kh = kecil
    bx, by, bw, bh = besar
    ix = max(0, min(kx + kw, bx + bw) - max(kx, bx))
    iy = max(0, min(ky + kh, by + bh) - max(ky, by))
    return ix * iy >= batas * kw * kh


def _buang_duplikat(kandidat, batas=0.5):
    """1) Satu plat menghasilkan kontur luar dan dalam: simpan yang skornya tertinggi.
    2) Blok teks yang berada DI DALAM bingkai plat dibuang, supaya crop memuat seluruh plat
       (bukan hanya sebagian karakter)."""
    hasil = []
    for k in kandidat:
        if any(_iou(k["box"], h["box"]) >= batas for h in hasil):
            continue
        if any(o is not k and _didalam(k["box"], o["box"]) and o["box"] != k["box"]
               and o["box"][2] * o["box"][3] > k["box"][2] * k["box"][3] for o in kandidat):
            continue
        hasil.append(k)
    return hasil


def find_plate_candidates(img, debug=None):
    """Return daftar kandidat (terbaik dulu): dict(box, score, aspect, crop).
    `debug` (dict) diisi gambar tiap langkah bila diberikan."""
    img = _limit_width(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, PLATE_BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, *PLATE_CANNY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, PLATE_CLOSE_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # RETR_LIST (bukan EXTERNAL): plat berada DI DALAM kontur badan mobil, jadi kontur dalam harus ikut dicari
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    img_area = float(H * W)
    kandidat = []
    for c in contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if rw < 1 or rh < 1:
            continue
        w, h = max(rw, rh), min(rw, rh)
        aspect = w / h
        rect_area = w * h
        if not (PLATE_ASPECT_RANGE[0] <= aspect <= PLATE_ASPECT_RANGE[1]):
            continue
        if not (PLATE_AREA_RATIO[0] <= rect_area / img_area <= PLATE_AREA_RATIO[1]):
            continue
        rectangularity = cv2.contourArea(c) / rect_area
        if rectangularity < PLATE_MIN_RECTANGULARITY:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # plat berisi banyak karakter -> kepadatan tepi tinggi; jendela/kaca kosong rendah
        density = float(np.count_nonzero(edges[y:y + bh, x:x + bw])) / (bw * bh)
        mx, my = int(PLATE_CROP_MARGIN * bw), int(PLATE_CROP_MARGIN * bh)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + bw + mx), min(H, y + bh + my)
        kandidat.append({
            "box": (x0, y0, x1 - x0, y1 - y0),
            "score": rectangularity * density,
            "aspect": round(aspect, 2),
            "crop": img[y0:y1, x0:x1].copy(),
        })

    kandidat.sort(key=lambda k: k["score"], reverse=True)
    kandidat = _buang_duplikat(kandidat)[:PLATE_MAX_CANDIDATES]

    if debug is not None:
        overlay = img.copy()
        for i, k in enumerate(kandidat):
            x, y, w, h = k["box"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0) if i == 0 else (0, 165, 255), 2)
        debug.update({"01_asli": img, "02_grayscale": gray, "03_gaussian_blur": blur,
                      "04_canny": edges, "05_closing": closed, "06_kandidat": overlay})
        if kandidat:
            debug["07_crop_plat"] = kandidat[0]["crop"]
    return kandidat
