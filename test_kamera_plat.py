"""
Test pembacaan plat nomor real-time dari kamera - TANPA LAG.
Gaya seperti recognizer.py tapi OCR berjalan di thread background
sehingga tampilan kamera tetap mulus 30fps.

Cara pakai:
    python test_kamera_plat.py              # webcam default
    python test_kamera_plat.py --cam 1      # webcam index lain
    python test_kamera_plat.py --ip http://192.168.X.X:8080/shot.jpg

Kontrol:
    Q / ESC  -> keluar
    +  / -   -> percepat / perlambat frekuensi OCR
"""
import argparse
import time
import threading
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.vision.plate_reader import read_plate, save_debug

DEBUG_DIR = Path("data/debug")
CONFIRM_COUNT = 3


# ---------------------------------------------------------------------------
# Buka kamera
# ---------------------------------------------------------------------------
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


def baca_ipcam(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[IPCam] {e}"); return None


# ---------------------------------------------------------------------------
# Thread OCR — proses berjalan di background, UI tidak freeze
# ---------------------------------------------------------------------------
class OCRThread:
    """Terima frame snapshot, jalankan read_plate di thread terpisah,
    simpan hasil terbaru. Main loop hanya baca hasil, tidak perlu tunggu."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._input  = None    # frame yang perlu diproses
        self._result = None    # hasil terakhir
        self._busy   = False
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def kirim(self, frame):
        """Kirim frame baru ke thread. Diabaikan bila thread masih sibuk."""
        with self._lock:
            if not self._busy:
                self._input = frame.copy()

    def ambil(self):
        """Ambil hasil terbaru (None bila belum ada)."""
        with self._lock:
            return self._result

    def sibuk(self):
        with self._lock:
            return self._busy

    def _loop(self):
        while True:
            frame = None
            with self._lock:
                if self._input is not None:
                    frame = self._input
                    self._input = None
                    self._busy  = True
            if frame is not None:
                dbg   = {}
                hasil = read_plate(frame, dbg, fast=True)

                # Auto-debug ke disk saja
                if (hasil.get("plate_text")
                        and hasil.get("plate_confidence", 0) >= 0.35
                        and dbg):
                    nama = time.strftime("plat_%H%M%S")
                    save_debug(dbg, DEBUG_DIR / nama)
                    print(f"[DEBUG] Disimpan -> data/debug/{nama}/")
                with self._lock:
                    self._result = hasil
                    self._busy   = False
            else:
                time.sleep(0.005)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--ip",  type=str, default=None)
    args = parser.parse_args()

    if args.ip:
        print(f"[INFO] IP Camera: {args.ip}")
        cap = None
    else:
        cap = buka_kamera(args.cam)
        if cap is None:
            print("[ERROR] Kamera tidak bisa dibuka. Coba --cam 1 atau --ip <url>")
            return

    ocr   = OCRThread()

    history          = deque(maxlen=8)
    confirmed_text   = None
    confirmed_conf   = 0.0
    waktu_konfirmasi = 0.0

    last_box   = None
    last_text  = None
    last_conf  = 0.0
    last_valid = False
    last_debug_text = None
    last_debug_time = 0.0

    # Interval kirim frame ke OCR (detik). Tesseract butuh ~0.5-2 detik.
    OCR_INTERVAL = 1.0
    last_send    = 0.0

    print("Kamera aktif. Tekan Q untuk keluar, +/- ubah interval OCR.")

    try:
        while True:
            # ── Baca frame (selalu, tanpa tunggu OCR) ────────────────────────
            if args.ip:
                frame = baca_ipcam(args.ip)
                if frame is None:
                    time.sleep(0.05); continue
            else:
                try:
                    ok, frame = cap.read()
                except cv2.error:
                    continue
                if not ok or frame is None or frame.size == 0:
                    continue

            sekarang = time.time()

            # ── Kirim ke OCR thread setiap OCR_INTERVAL detik ────────────────
            if sekarang - last_send >= OCR_INTERVAL and not ocr.sibuk():
                ocr.kirim(frame)
                last_send = sekarang

            # ── Ambil hasil terbaru dari OCR thread (tidak blocking) ──────────
            hasil = ocr.ambil()
            if hasil is not None:
                last_box   = hasil.get("box")
                last_text  = hasil.get("plate_text")
                last_conf  = hasil.get("plate_confidence") or 0.0
                last_valid = hasil.get("plate_valid", False)

                # Stabilisasi konfirmasi
                if last_valid:
                    history.append(last_text)
                    counter = Counter(t for t in history if t)
                    if counter:
                        dominan, jumlah = counter.most_common(1)[0]
                        if jumlah >= CONFIRM_COUNT and dominan != confirmed_text:
                            confirmed_text   = dominan
                            confirmed_conf   = last_conf
                            waktu_konfirmasi = sekarang
                            print(f"  [OK] TERKONFIRMASI: {confirmed_text} "
                                  f"(conf {confirmed_conf:.2f})")
                else:
                    history.append(None)

            # ── Overlay di frame (gaya recognizer.py) ────────────────────────
            masih_konfirmasi = (confirmed_text
                                and sekarang - waktu_konfirmasi < 5.0)

            if last_box is not None:
                x, y, w, h = [int(v) for v in last_box]

                if masih_konfirmasi:
                    warna = (0, 255, 0)
                    label = f"PLAT: {confirmed_text} ({confirmed_conf:.2f})"
                elif last_valid:
                    warna = (0, 200, 255)
                    label = f"{last_text} ({last_conf:.2f})"
                else:
                    warna = (0, 100, 255)
                    label = last_text or "Tidak terbaca"

                cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 2)
                cv2.putText(frame, label,
                            (x, max(y - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
            else:
                cv2.putText(frame, "Tidak ada plat terdeteksi",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Status OCR di pojok kanan atas
            status_ocr = "OCR: proses..." if ocr.sibuk() else f"OCR: interval {OCR_INTERVAL:.1f}s"
            cv2.putText(frame, status_ocr,
                        (frame.shape[1] - 220, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Petunjuk bawah
            cv2.putText(frame,
                        f"[Q]=Keluar  [+/-]=Interval OCR ({OCR_INTERVAL:.1f}s)",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("ParkVisi - Deteksi Plat Nomor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('+'), ord('=')):
                OCR_INTERVAL = max(0.3, round(OCR_INTERVAL - 0.2, 1))
                print(f"[INFO] Interval OCR: {OCR_INTERVAL}s")
            elif key in (ord('-'), ord('_')):
                OCR_INTERVAL = min(5.0, round(OCR_INTERVAL + 0.2, 1))
                print(f"[INFO] Interval OCR: {OCR_INTERVAL}s")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
