"""
Verifikasi akses parkir real-time dari satu kamera — pakai Laravel API.

Alur:
  1. Deteksi wajah (Haar Cascade + LBPH)   ─┐  lokal, cepat
  2. Baca plat nomor (OCR Tesseract)         ─┤  lokal, thread background
  3. Kirim member_id + plate_text ke Laravel ─┘  → dapat status izin/tolak

Cara pakai:
    python test_akses_parkir.py                           # webcam 0, Laravel localhost:8000
    python test_akses_parkir.py --cam 1                   # webcam lain
    python test_akses_parkir.py --ip http://IP:8080/shot.jpg
    python test_akses_parkir.py --laravel http://IP:8000  # Laravel di server lain

Kontrol:
    Q / ESC  -> keluar
    R        -> reset / hapus hasil
    + / -    -> percepat / perlambat interval pemrosesan
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

# ─── Path setup ─────────────────────────────────────────────────────────────
import sys
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from app.config import UNKNOWN_THRESHOLD
from app.intelligence import lbph
from app.vision import face_detector, face_preprocess
from app.vision.plate_reader import read_plate, save_debug

DEBUG_DIR = Path("data/debug")


# ─── Laravel API helper ───────────────────────────────────────────────────────
class LaravelAPI:
    """Kirim member_id + plate_text ke endpoint Laravel /api/verifikasi-akses."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.url = base_url.rstrip("/") + "/api/verifikasi-akses"

    def verifikasi(self, member_id: int | None, plate_text: str | None) -> dict:
        """Return dict hasil dari Laravel, atau dict error bila gagal."""
        payload = json.dumps({
            "member_id":  member_id,
            "plate_text": plate_text,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return {"status": "error", "keterangan": f"HTTP {e.code}: {body[:120]}"}
        except Exception as e:
            return {"status": "error", "keterangan": f"Laravel tidak terjangkau: {e}"}


# ─── Buka kamera ─────────────────────────────────────────────────────────────
def buka_kamera(index: int):
    backends = [
        (cv2.CAP_MSMF,  "MSMF",  False),
        (cv2.CAP_ANY,   "ANY",   True),
        (cv2.CAP_DSHOW, "DSHOW", True),
        (-1,            "DEF",   True),
    ]
    for b, nm, set_res in backends:
        try:
            cap = cv2.VideoCapture(index, b) if b != -1 else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release(); continue
            ok, fr = cap.read()
            if not ok or fr is None or fr.size == 0:
                cap.release(); continue
            if set_res:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok2, fr2 = cap.read()
                if not ok2 or fr2 is None or fr2.size == 0:
                    cap.release(); continue
            print(f"[INFO] Kamera berhasil dibuka (backend {nm})")
            return cap
        except Exception as e:
            print(f"[WARN] {nm}: {e}")
    return None


def baca_ipcam(url: str):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[IPCam] {e}")
        return None


# ─── Thread utama (wajah + plat + API) ───────────────────────────────────────
class VerifikasiThread:
    """
    Background thread yang menangani:
      1. Deteksi & pengenalan wajah (LBPH, lokal)
      2. OCR plat nomor (Tesseract, lokal)
      3. POST ke Laravel API → status akses
    Main loop tidak pernah diblokir.
    """

    def __init__(self, api: LaravelAPI):
        self._api    = api
        self._lock   = threading.Lock()
        self._input  = None
        self._result = None
        self._busy   = False
        threading.Thread(target=self._loop, daemon=True).start()

    def kirim(self, frame):
        with self._lock:
            if not self._busy:
                self._input = frame.copy()

    def ambil(self):
        with self._lock:
            return self._result

    def sibuk(self):
        with self._lock:
            return self._busy

    # ── Loop internal ────────────────────────────────────────────────────────
    def _loop(self):
        while True:
            frame = None
            with self._lock:
                if self._input is not None:
                    frame      = self._input
                    self._input = None
                    self._busy  = True
            if frame is not None:
                hasil = self._proses(frame)
                with self._lock:
                    self._result = hasil
                    self._busy   = False
            else:
                time.sleep(0.01)

    def _proses(self, frame) -> dict:
        """Jalankan deteksi wajah + OCR plat, lalu panggil Laravel API."""
        hasil = {
            # Wajah (lokal)
            "wajah_terdeteksi": False,
            "wajah_box":        None,
            "wajah_member_id":  None,
            "wajah_distance":   None,
            "wajah_cocok":      False,
            # Plat (lokal)
            "plat_terdeteksi":  False,
            "plat_box":         None,
            "plat_text":        None,
            "plat_confidence":  0.0,
            "plat_valid":       False,
            # Dari Laravel API
            "status":           "menunggu",
            "keterangan":       "Arahkan wajah dan plat ke kamera",
            "cocok":            False,
            "member_name":      None,
            "plat_pemilik":     None,
        }

        gray = face_preprocess.to_gray(frame)

        # ── 1. Deteksi wajah ─────────────────────────────────────────────
        wajah_box = face_detector.detect_face(gray)
        if wajah_box is not None:
            hasil["wajah_terdeteksi"] = True
            hasil["wajah_box"]        = wajah_box
            face_crop = face_preprocess.normalize(
                face_preprocess.crop_face(gray, wajah_box)
            )
            pred = lbph.predict(face_crop)
            if pred is not None:
                mid, dist = pred
                if dist <= UNKNOWN_THRESHOLD:
                    hasil["wajah_cocok"]     = True
                    hasil["wajah_member_id"] = int(mid)
                    hasil["wajah_distance"]  = round(dist, 1)

        # ── 2. OCR plat ───────────────────────────────────────────────────
        dbg      = {}
        plat_res = read_plate(frame, dbg, fast=True)
        if plat_res.get("plate_text"):
            hasil["plat_terdeteksi"] = True
            hasil["plat_text"]       = plat_res["plate_text"]
            hasil["plat_confidence"] = plat_res.get("plate_confidence") or 0.0
            hasil["plat_valid"]      = plat_res.get("plate_valid", False)
            hasil["plat_box"]        = plat_res.get("box")

            if hasil["plat_valid"] and hasil["plat_confidence"] >= 0.35 and dbg:
                nama = time.strftime("akses_%H%M%S")
                save_debug(dbg, DEBUG_DIR / nama)

        # ── 3. Panggil Laravel API ────────────────────────────────────────
        member_id  = hasil["wajah_member_id"] if hasil["wajah_cocok"] else None
        plate_text = hasil["plat_text"]       if hasil["plat_valid"]  else None

        api_resp = self._api.verifikasi(member_id, plate_text)

        hasil["status"]      = api_resp.get("status",     "menunggu")
        hasil["keterangan"]  = api_resp.get("keterangan", "")
        hasil["cocok"]       = api_resp.get("cocok",      False)
        hasil["plat_pemilik"] = api_resp.get("plat_pemilik")

        member_info = api_resp.get("member")
        if member_info:
            hasil["member_name"] = member_info.get("name")

        return hasil


# ─── Overlay ─────────────────────────────────────────────────────────────────
HIJAU  = (0,  220,  60)
MERAH  = (0,   40, 230)
ORANYE = (0,  165, 255)
KUNING = (0,  210, 210)
ABU    = (180, 180, 180)
PUTIH  = (255, 255, 255)

STATUS_WARNA = {
    "izin":     HIJAU,
    "tolak":    MERAH,
    "mismatch": KUNING,
    "menunggu": ORANYE,
    "error":    (100, 100, 255),
}

LABEL_STATUS = {
    "izin":     "✓  AKSES DITERIMA",
    "tolak":    "✗  AKSES DITOLAK",
    "mismatch": "⚠  WAJAH & PLAT TIDAK COCOK",
    "menunggu": "●  MENUNGGU DATA...",
    "error":    "!  ERROR API",
}


def gambar_overlay(frame, hasil: dict | None, sibuk: bool, interval: float):
    H, W = frame.shape[:2]

    # Strip atas semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if hasil is None:
        cv2.putText(frame, "Memuat model...", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANYE, 2)
        return

    status      = hasil.get("status", "menunggu")
    warna       = STATUS_WARNA.get(status, ABU)
    label_st    = LABEL_STATUS.get(status, status.upper())
    keterangan  = hasil.get("keterangan", "")

    # Status utama di strip atas
    cv2.putText(frame, label_st, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, warna, 2)

    # Keterangan di baris bawah strip
    if keterangan:
        # potong bila terlalu panjang supaya muat layar
        max_kar = max(40, W // 12)
        if len(keterangan) > max_kar:
            keterangan = keterangan[:max_kar - 1] + "…"
        cv2.putText(frame, keterangan, (12, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, PUTIH, 1)

    # ── Bounding box WAJAH ────────────────────────────────────────────────
    box_w = hasil.get("wajah_box")
    if box_w is not None:
        x, y, w, h = map(int, box_w)
        col_w  = HIJAU if hasil.get("wajah_cocok") else ORANYE
        cv2.rectangle(frame, (x, y), (x + w, y + h), col_w, 2)

        if hasil.get("wajah_cocok"):
            nama = hasil.get("member_name") or f"ID {hasil['wajah_member_id']}"
            dist = hasil.get("wajah_distance", "")
            lbl  = f"{nama}  d={dist}" if dist else nama
        else:
            lbl = "Asing / tidak dikenal"

        cv2.putText(frame, lbl, (x, max(y - 8, 65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, col_w, 2)
    elif not hasil.get("wajah_terdeteksi"):
        cv2.putText(frame, "Wajah tidak terdeteksi",
                    (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ORANYE, 1)

    # ── Bounding box PLAT ─────────────────────────────────────────────────
    box_p      = hasil.get("plat_box")
    plat_text  = hasil.get("plat_text")
    if box_p is not None and plat_text:
        x, y, w, h = map(int, box_p)
        col_p = HIJAU if hasil.get("plat_valid") else ORANYE
        cv2.rectangle(frame, (x, y), (x + w, y + h), col_p, 2)
        conf  = hasil.get("plat_confidence", 0.0)
        cv2.putText(frame, f"PLAT: {plat_text}  ({conf:.2f})",
                    (x, max(y - 8, 65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_p, 2)
    elif not plat_text:
        cy = 85 if hasil.get("wajah_terdeteksi") else 110
        cv2.putText(frame, "Plat belum terbaca",
                    (12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ORANYE, 1)

    # ── Info pemilik plat (dari API) jika ada mismatch ────────────────────
    plat_pemilik = hasil.get("plat_pemilik")
    if plat_pemilik and status == "mismatch":
        info = f"Plat terdaftar atas: {plat_pemilik.get('member_name', '-')}"
        cv2.putText(frame, info, (12, H - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, KUNING, 1)

    # ── Status proses & panduan ───────────────────────────────────────────
    teks_kanan = "Proses..." if sibuk else f"Interval {interval:.1f}s"
    (tw, _), _ = cv2.getTextSize(teks_kanan, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, teks_kanan, (W - tw - 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ABU, 1)

    cv2.putText(frame, "[Q/ESC]=Keluar  [R]=Reset  [+/-]=Interval",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ABU, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verifikasi akses parkir: wajah + plat → Laravel API"
    )
    parser.add_argument("--cam",     type=int, default=0,
                        help="Index webcam (default 0)")
    parser.add_argument("--ip",      type=str, default=None,
                        help="URL IP Camera snapshot")
    parser.add_argument("--laravel", type=str, default="http://127.0.0.1:8000",
                        help="URL base Laravel (default http://127.0.0.1:8000)")
    args = parser.parse_args()

    # ── Muat model LBPH ───────────────────────────────────────────────────
    print("[INFO] Memuat model LBPH...")
    if not lbph.load_model():
        print("[WARN] Model LBPH tidak ditemukan — deteksi wajah berjalan tanpa pengenalan.")

    # ── Siapkan API client ────────────────────────────────────────────────
    api = LaravelAPI(args.laravel)
    print(f"[INFO] Laravel API: {api.url}")

    # Coba ping API
    ping = api.verifikasi(None, None)
    if ping.get("status") == "error":
        print(f"[WARN] {ping['keterangan']}")
        print("       Pastikan Laravel sedang berjalan (php artisan serve).")
    else:
        print(f"[INFO] Laravel API terhubung ✓  (status={ping['status']})")

    # ── Buka kamera ───────────────────────────────────────────────────────
    if args.ip:
        print(f"[INFO] Menggunakan IP Camera: {args.ip}")
        cap = None
    else:
        print(f"[INFO] Membuka webcam index {args.cam}...")
        cap = buka_kamera(args.cam)
        if cap is None:
            print("[ERROR] Kamera tidak bisa dibuka. Coba --cam 1 atau --ip <url>")
            return

    # ── Thread verifikasi ─────────────────────────────────────────────────
    veri = VerifikasiThread(api)

    # Stabilisasi teks plat
    plat_history    = deque(maxlen=8)
    confirmed_plat  = None
    plat_conf_waktu = 0.0
    CONFIRM_COUNT   = 3

    INTERVAL  = 1.2
    last_send = 0.0
    last_log  = ""   # hindari cetak status yang sama berulang

    print("\nSistem aktif. Q=Keluar | R=Reset | +/-=Interval\n")

    try:
        while True:
            # ── Ambil frame ───────────────────────────────────────────────
            if args.ip:
                frame = baca_ipcam(args.ip)
                if frame is None:
                    time.sleep(0.05)
                    continue
            else:
                try:
                    ok, frame = cap.read()
                except cv2.error:
                    continue
                if not ok or frame is None or frame.size == 0:
                    continue

            sekarang = time.time()

            # ── Kirim ke thread (non-blocking) ────────────────────────────
            if sekarang - last_send >= INTERVAL and not veri.sibuk():
                veri.kirim(frame)
                last_send = sekarang

            # ── Ambil hasil ───────────────────────────────────────────────
            hasil = veri.ambil()

            if hasil:
                # Stabilisasi plat
                if hasil.get("plat_valid"):
                    plat_history.append(hasil["plat_text"])
                    ctr = Counter(t for t in plat_history if t)
                    if ctr:
                        dom, jml = ctr.most_common(1)[0]
                        if jml >= CONFIRM_COUNT and dom != confirmed_plat:
                            confirmed_plat  = dom
                            plat_conf_waktu = sekarang
                            print(f"  [PLAT] Terkonfirmasi: {confirmed_plat}")
                else:
                    plat_history.append(None)

                # Log ke konsol bila status berubah
                log_baru = f"{hasil['status']}|{hasil['keterangan']}"
                if log_baru != last_log:
                    st = hasil["status"]
                    if st in ("izin", "tolak", "mismatch", "error"):
                        ikon = {"izin": "✓", "tolak": "✗",
                                "mismatch": "⚠", "error": "!"}[st]
                        print(f"  [{ikon}] {hasil['keterangan']}")
                    last_log = log_baru

            # ── Gambar overlay ────────────────────────────────────────────
            gambar_overlay(frame, hasil, veri.sibuk(), INTERVAL)
            cv2.imshow("ParkVisi – Verifikasi Akses Parkir", frame)

            # ── Keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                plat_history.clear()
                confirmed_plat  = None
                plat_conf_waktu = 0.0
                last_log        = ""
                print("[INFO] Reset.")
            elif key in (ord("+"), ord("=")):
                INTERVAL = max(0.5, round(INTERVAL - 0.2, 1))
                print(f"[INFO] Interval: {INTERVAL}s")
            elif key in (ord("-"), ord("_")):
                INTERVAL = min(5.0, round(INTERVAL + 0.2, 1))
                print(f"[INFO] Interval: {INTERVAL}s")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    main()
