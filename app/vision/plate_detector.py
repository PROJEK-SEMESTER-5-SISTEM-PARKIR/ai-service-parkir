"""Lokasi plat nomor dengan teknik konvensional.

Dua sumber kandidat:
  1) Tepi : grayscale -> CLAHE -> Gaussian Blur -> Canny -> closing -> kontur -> filter geometri
  2) Blob : adaptive threshold -> komponen terhubung -> deretan blob karakter sejajar
Semua kandidat diluruskan (perspektif) lalu dinilai dari ISINYA (jumlah karakter di dalam crop),
bukan hanya bentuknya, sehingga kaca depan / gril tidak mengalahkan plat.
"""
import cv2
import numpy as np

from app.config import (
    PLATE_ASPECT_RANGE, PLATE_AREA_RATIO, PLATE_BLUR_KERNEL, PLATE_CANNY,
    PLATE_CLOSE_KERNEL, PLATE_CLOSE_KERNEL_KECIL, PLATE_CROP_MARGIN,
    PLATE_MAX_CANDIDATES, PLATE_MAX_WIDTH, PLATE_MIN_RECTANGULARITY,
)
from app.vision import plate_segment

ASPECT_IDEAL = 3.0          # rasio lebar:tinggi plat untuk pembobotan
TOP_TEPI = 20               # kandidat tepi terbaik yang dinilai isinya
TOP_BLOB = 15               # kandidat blob terbaik yang dinilai isinya
IOU_NMS = 0.4

# Plat nomor Indonesia umumnya berada di bagian bawah kendaraan.
# Kandidat yang y-centernya di atas batas ini diberi penalty skor.
Y_CENTER_MIN_RATIO = 0.40   # plat jarang ada di atas 40% tinggi gambar
Y_CENTER_MAX_RATIO = 0.97   # plat jarang ada di bawah 3% dari dasar gambar

# Area kandidat relatif luas gambar.
# > PLATE_AREA_SOFT_MAX : kemungkinan bukan plat (kaca, gril) -> skor dikurangi
# > PLATE_AREA_HARD_MAX : pasti bukan plat (objek terlalu besar) -> dibuang
# Catatan: foto close-up plat bisa > 20%, jadi hard max harus cukup besar.
PLATE_AREA_SOFT_MAX = 0.15  # di atas ini diberi penalty skor 0.4
PLATE_AREA_HARD_MAX = 0.55  # di atas ini pasti bukan plat (> setengah frame)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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


def _bobot_aspek(aspect):
    return max(0.3, 1.0 - abs(aspect - ASPECT_IDEAL) / 3.0)


def _urut_titik(pts):
    """Urutkan 4 titik: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def _luruskan(img, rect):
    """Crop persegi panjang miring (minAreaRect) menjadi gambar tegak dengan perspective warp."""
    box = cv2.boxPoints(rect)
    pts = _urut_titik(box)
    pusat = box.mean(axis=0)
    pts = (pusat + (pts - pusat) * (1 + 2 * PLATE_CROP_MARGIN)).astype(np.float32)
    lebar = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    tinggi = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    if lebar < 8 or tinggi < 4:
        return None
    dst = np.array([[0, 0], [lebar - 1, 0], [lebar - 1, tinggi - 1], [0, tinggi - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(img, M, (lebar, tinggi), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    if tinggi > lebar:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def _jumlah_karakter(gray_crop):
    r = plate_segment.cari_karakter(gray_crop)
    return r["n"] if r else 0


# ---------------------------------------------------------------------------
# Sumber 1: tepi + kontur
# ---------------------------------------------------------------------------
def _dari_tepi(img, gray_c, H, W):
    blur = cv2.GaussianBlur(gray_c, PLATE_BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, *PLATE_CANNY)
    img_area = float(H * W)
    calon, closed_pertama = [], None

    for kern in (PLATE_CLOSE_KERNEL, PLATE_CLOSE_KERNEL_KECIL):
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kern))
        if closed_pertama is None:
            closed_pertama = closed
        # RETR_LIST: plat berada DI DALAM kontur badan mobil, kontur dalam harus ikut dicari
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw < 1 or rh < 1:
                continue
            w, h = max(rw, rh), min(rw, rh)
            aspect, rect_area = w / h, w * h
            if not (PLATE_ASPECT_RANGE[0] <= aspect <= PLATE_ASPECT_RANGE[1]):
                continue
            if not (PLATE_AREA_RATIO[0] <= rect_area / img_area <= PLATE_AREA_RATIO[1]):
                continue
            rectangularity = cv2.contourArea(c) / rect_area
            if rectangularity < PLATE_MIN_RECTANGULARITY:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter keras: buang kandidat terlalu besar (kaca depan penuh, dll.)
            if rect_area / img_area > PLATE_AREA_HARD_MAX:
                continue
            # Soft penalty untuk kandidat agak besar (kemungkinan bukan plat)
            area_penalty = 0.4 if rect_area / img_area > PLATE_AREA_SOFT_MAX else 1.0
            # Skor posisi vertikal: beri penalty bila y-center tidak di zona plat
            cy_rel = (y + bh / 2.0) / H
            if cy_rel < Y_CENTER_MIN_RATIO or cy_rel > Y_CENTER_MAX_RATIO:
                posisi_ok = 0.5   # penalty: masih dicoba tapi skornya rendah
            else:
                posisi_ok = 1.0
            density = float(np.count_nonzero(edges[y:y + bh, x:x + bw])) / (bw * bh)
            calon.append({"rect": rect, "aspect": aspect, "rect_ratio": min(1.0, rectangularity),
                          "pra": rectangularity * density * _bobot_aspek(aspect) * posisi_ok * area_penalty})

    calon.sort(key=lambda k: k["pra"], reverse=True)
    hasil = []
    for c in calon[:TOP_TEPI]:
        crop = _luruskan(img, c["rect"])
        if crop is None:
            continue
        pts = cv2.boxPoints(c["rect"])
        x0, y0 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        x1, y1 = int(min(W, pts[:, 0].max())), int(min(H, pts[:, 1].max()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        hasil.append({"box": (x0, y0, x1 - x0, y1 - y0), "crop": crop, "aspect": c["aspect"],
                      "bobot": _bobot_aspek(c["aspect"]),
                      "kualitas": 0.6 + 0.4 * c["rect_ratio"], "sumber": "tepi"})
    return edges, closed_pertama, blur, hasil


# ---------------------------------------------------------------------------
# Sumber 2: deretan blob karakter
# ---------------------------------------------------------------------------
def _dari_karakter(img, gray_c, H, W):
    halus = cv2.GaussianBlur(gray_c, (3, 3), 0)
    blok = max(31, (min(H, W) // 10) | 1)
    adaptif = cv2.adaptiveThreshold(halus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blok, 10)
    calon = []
    for biner in (adaptif, cv2.bitwise_not(adaptif)):
        _, _, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
        x, y, w, h, luas = stats[1:].T
        rasio = w / np.maximum(h, 1)
        ok = ((h >= 0.02 * H) & (h <= 0.5 * H) & (w >= 2) & (rasio >= 0.1) & (rasio <= 1.1)
              & (luas >= 20) & (luas >= 0.2 * w * h))
        idx = np.where(ok)[0]
        if len(idx) < 4:
            continue
        if len(idx) > 600:
            idx = idx[np.argsort(-luas[idx])[:600]]
        arr = np.stack([x[idx], y[idx], w[idx], h[idx]], axis=1).astype(float)
        for run in plate_segment.kelompok_baris(arr):
            if 4 <= len(run) <= 10:
                calon.append(arr[run])

    calon.sort(key=len, reverse=True)
    hasil = []
    for g in calon[:TOP_BLOB]:
        tinggi = g[:, 3]
        h_med = float(np.median(tinggi))
        x0 = int(max(0, g[:, 0].min() - 0.6 * h_med))
        x1 = int(min(W, (g[:, 0] + g[:, 2]).max() + 0.6 * h_med))
        y0 = int(max(0, g[:, 1].min() - 0.5 * h_med))
        y1 = int(min(H, (g[:, 1] + g[:, 3]).max() + 0.5 * h_med))
        bw, bh = x1 - x0, y1 - y0
        if bw < 8 or bh < 4:
            continue
        # Filter keras area
        if bw * bh / float(H * W) > PLATE_AREA_HARD_MAX:
            continue
        aspect = bw / bh
        if not (1.5 <= aspect <= 7.0):
            continue
        # Soft penalty dan posisi vertikal untuk blob
        area_rel = bw * bh / float(H * W)
        area_penalty = 0.4 if area_rel > PLATE_AREA_SOFT_MAX else 1.0
        cy_rel = (y0 + bh / 2.0) / H
        posisi_ok = 1.0 if Y_CENTER_MIN_RATIO <= cy_rel <= Y_CENTER_MAX_RATIO else 0.5
        seragam = max(0.0, 1.0 - 2.0 * float(tinggi.std() / tinggi.mean()))
        hasil.append({"box": (x0, y0, bw, bh), "crop": img[y0:y1, x0:x1].copy(), "aspect": aspect,
                      "bobot": _bobot_aspek(aspect) * posisi_ok * area_penalty,
                      "kualitas": 0.6 + 0.4 * seragam, "sumber": "blob"})
    return hasil


# ---------------------------------------------------------------------------
def find_plate_candidates(img, debug=None):
    """Return daftar kandidat (terbaik dulu): dict(box, score, aspect, crop, nchar, sumber).
    `debug` (dict) diisi gambar tiap langkah bila diberikan."""
    img = _limit_width(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gray_c = _clahe.apply(gray)          # bantu foto gelap

    edges, closed, blur, kandidat = _dari_tepi(img, gray_c, H, W)
    kandidat += _dari_karakter(img, gray_c, H, W)

    # Nilai ISI kandidat: jumlah karakter di dalam crop
    for k in kandidat:
        nchar = _jumlah_karakter(cv2.cvtColor(k["crop"], cv2.COLOR_BGR2GRAY))
        k["nchar"] = nchar
        faktor = 1.0 + min(nchar, 8) / 4 if 4 <= nchar <= 10 else 0.3
        k["score"] = faktor * k["bobot"] * k["kualitas"]

    kandidat.sort(key=lambda k: k["score"], reverse=True)
    terpilih = []
    for k in kandidat:
        if any(_iou(k["box"], t["box"]) >= IOU_NMS for t in terpilih):
            continue
        terpilih.append(k)
        if len(terpilih) >= PLATE_MAX_CANDIDATES:
            break

    if debug is not None:
        overlay = img.copy()
        for i, k in enumerate(terpilih):
            x, y, w, h = k["box"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0) if i == 0 else (0, 165, 255), 2)
            cv2.putText(overlay, f'{k["sumber"]} n={k["nchar"]}', (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        debug.update({"01_asli": img, "02_grayscale": gray, "03_gaussian_blur": blur,
                      "04_canny": edges, "05_closing": closed, "06_kandidat": overlay})
        if terpilih:
            debug["07_crop_plat"] = terpilih[0]["crop"]
    return terpilih