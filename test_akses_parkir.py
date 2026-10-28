"""
Verifikasi Akses Parkir Terpadu (Kamera AI + ESP32 IoT Pintu Masuk & Keluar).

Fitur:
  - Berjalan terus-menerus (otomatis) tanpa perlu dijalankan berulang kali.
  - Mendukung Pintu Masuk (gate_masuk) dan Pintu Keluar (gate_keluar).
  - Ketika sensor ultrasonik ESP32 mendeteksi kendaraan, kamera otomatis memindai wajah & plat.
  - Memvalidasi kecocokan data ke Laravel API & database.
  - Palang servo ESP32 otomatis terbuka jika cocok, atau ditolak jika tidak sesuai.
  - Setelah kendaraan lewat, sistem otomatis kembali ke status SIAGA untuk kendaraan berikutnya.

Kontrol:
  - Secara default berjalan dalam mode Web Streaming (tanpa popup window OpenCV/IDLE).
  - Hasil pemindaian langsung tampil di Web Admin: http://localhost:8000/parkir/kamera
  - Jika ingin menampilkan popup window OpenCV, gunakan opsi: --window
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


# ─── Server Stream MJPEG (untuk ditampilkan di web Laravel) ──────────────────
class StreamServer:
    """Menyiarkan frame terakhir (sudah ada kotak wajah/plat) lewat HTTP."""
    _lock = threading.Lock()
    _jpeg = None

    @classmethod
    def update(cls, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with cls._lock:
                cls._jpeg = buf.tobytes()

    @classmethod
    def get(cls):
        with cls._lock:
            return cls._jpeg

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # matikan log di terminal
            pass

        def do_GET(self):
            if not self.path.startswith("/stream"):
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    jpg = StreamServer.get()
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.06)  # ± 15 fps
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    @classmethod
    def start(cls, port: int):
        srv = ThreadingHTTPServer(("0.0.0.0", port), cls.Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[INFO] Stream kamera aktif: http://127.0.0.1:{port}/stream")


# ─── Path setup ─────────────────────────────────────────────────────────────
import sys
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from app.config import UNKNOWN_THRESHOLD
from app.intelligence import lbph
from app.vision import face_detector, face_preprocess
from app.vision.plate_reader import read_plate, save_debug

DEBUG_DIR = Path("data/debug")


# ─── Laravel API Client ─────────────────────────────────────────────────────
class LaravelGateAPI:
    """Komunikasi dengan Laravel untuk status gerbang dan verifikasi akses."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, data: dict | None = None, method: str = "GET") -> dict:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_json = json.loads(e.read().decode("utf-8"))
                return {"error": True, "code": e.code, "message": err_json.get("message", str(e))}
            except Exception:
                return {"error": True, "code": e.code, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Koneksi gagal ke {url}: {e}"}

    def cek_status_gate(self, gate: str | None = None) -> dict:
        """Cek apakah ada gerbang yang sedang meminta verifikasi."""
        endpoint = f"/api/parking/verification-status?gate={gate}" if gate else "/api/parking/verification-status"
        return self._request(endpoint)

    def trigger_manual(self, gate: str = "gate_masuk") -> dict:
        """Kirim trigger manual (simulasi sensor mendeteksi mobil)."""
        return self._request("/api/parking/trigger", {"gate": gate}, method="POST")

    def reset_gate(self, gate: str = "gate_masuk") -> dict:
        """Reset state gerbang ke SIAGA."""
        return self._request("/api/parking/reset", {"gate": gate}, method="POST")

    def verifikasi_akses(self, member_id: int | None, plate_text: str | None,
                         distance: float | None = None, gate: str = "gate_masuk") -> dict:
        """Kirim hasil scan wajah dan plat nomor ke Laravel untuk diputuskan."""
        direction = "out" if "keluar" in gate.lower() else "in"
        payload = {
            "gate": gate,
            "direction": direction,
            "member_id": member_id,
            "plate_text": plate_text,
            "distance": distance,
            "update_gate": True,
        }
        return self._request("/api/verifikasi-akses", payload, method="POST")


# ─── Buka Kamera ────────────────────────────────────────────────────────────
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
                cap.release()
                continue
            ok, fr = cap.read()
            if not ok or fr is None or fr.size == 0:
                cap.release()
                continue
            if set_res:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok2, fr2 = cap.read()
                if not ok2 or fr2 is None or fr2.size == 0:
                    cap.release()
                    continue
            print(f"[INFO] Kamera terbuka (backend {nm})")
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


# ─── Thread Worker Pemindaian Wajah & Plat ──────────────────────────────────
class ScanWorker:
    """Thread background untuk mendeteksi wajah & membaca plat secara non-blocking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._input_frame = None
        self._hasil = None
        self._busy = False
        threading.Thread(target=self._loop, daemon=True).start()

    def kirim_frame(self, frame):
        with self._lock:
            if not self._busy:
                self._input_frame = frame.copy()

    def ambil_hasil(self):
        with self._lock:
            return self._hasil

    def is_busy(self):
        with self._lock:
            return self._busy

    def _loop(self):
        while True:
            frame = None
            with self._lock:
                if self._input_frame is not None:
                    frame = self._input_frame
                    self._input_frame = None
                    self._busy = True

            if frame is not None:
                hasil = self._proses_scan(frame)
                with self._lock:
                    self._hasil = hasil
                    self._busy = False
            else:
                time.sleep(0.01)

    def _proses_scan(self, frame) -> dict:
        res = {
            # Wajah
            "wajah_ada": False,
            "wajah_box": None,
            "wajah_id": None,
            "wajah_distance": None,
            "wajah_cocok": False,
            # Plat
            "plat_ada": False,
            "plat_box": None,
            "plat_text": None,
            "plat_conf": 0.0,
            "plat_valid": False,
        }

        gray = face_preprocess.to_gray(frame)

        # 1. Deteksi Wajah
        box_wajah = face_detector.detect_face(gray)
        if box_wajah is not None:
            res["wajah_ada"] = True
            res["wajah_box"] = box_wajah
            try:
                face_norm = face_preprocess.normalize(
                    face_preprocess.crop_face(gray, box_wajah)
                )
                pred = lbph.predict(face_norm)
                if pred is not None:
                    mid, dist = pred
                    res["wajah_distance"] = round(dist, 1)
                    if dist <= UNKNOWN_THRESHOLD:
                        res["wajah_cocok"] = True
                        res["wajah_id"] = int(mid)
            except Exception:
                pass

        # 2. Pembacaan Plat Nomor
        dbg = {}
        try:
            plate_res = read_plate(frame, dbg, fast=True)
            if plate_res.get("plate_text"):
                res["plat_ada"] = True
                res["plat_text"] = plate_res["plate_text"]
                res["plat_conf"] = plate_res.get("plate_confidence") or 0.0
                res["plat_valid"] = plate_res.get("plate_valid", False)
                res["plat_box"] = plate_res.get("box")

                if res["plat_valid"] and res["plat_conf"] >= 0.35 and dbg:
                    nama_dbg = time.strftime("akses_%H%M%S")
                    save_debug(dbg, DEBUG_DIR / nama_dbg)
        except Exception:
            pass

        return res


# ─── Warna Tampilan UI ──────────────────────────────────────────────────────
HIJAU     = (0, 230, 80)
MERAH     = (40, 40, 240)
ORANYE    = (0, 165, 255)
KUNING    = (0, 220, 240)
BIRU_CYAN = (240, 200, 0)
ABU       = (180, 180, 180)
PUTIH     = (255, 255, 255)
HITAM     = (20, 20, 20)


def gambar_interface(frame, state_sistem: str, target_gate: str, gate_info: dict,
                     scan_result: dict | None, verif_result: dict | None,
                     sisa_waktu: float, interval_detik: float):
    H, W = frame.shape[:2]

    nama_gate_teks = "PINTU KELUAR" if "keluar" in target_gate else "PINTU MASUK"

    # ── 1. Header Banner Atas ────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 70), HITAM, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    if state_sistem == "SIAGA":
        warna_header = BIRU_CYAN
        judul_header = "[SIAGA] MENUNGGU SENSOR KENDARAAN"
        sub_header = "Sensor ultrasonik, tombol web, atau deteksi kamera akan memicu sistem."
    elif state_sistem == "MEMPROSES":
        warna_header = ORANYE
        judul_header = f"[MEMPROSES {nama_gate_teks}] SEDANG MENCOCOKKAN ({sisa_waktu:.1f}s)"
        sub_header = f"Memindai wajah & plat nomor kendaraan ({target_gate})..."
    elif state_sistem == "HASIL":
        status_verif = verif_result.get("status") if verif_result else "menunggu"
        if status_verif == "izin":
            warna_header = HIJAU
            judul_header = f"✓  AKSES {nama_gate_teks} DITERIMA - PALANG DIBUKA"
        else:
            warna_header = MERAH
            judul_header = f"✗  AKSES {nama_gate_teks} DITOLAK - PALANG TERTUTUP"
        sub_header = verif_result.get("keterangan", "") if verif_result else ""
    else:
        warna_header = ABU
        judul_header = f"[{state_sistem}]"
        sub_header = ""

    cv2.putText(frame, judul_header, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, warna_header, 2)
    cv2.putText(frame, sub_header[:95], (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PUTIH, 1)

    # ── 2. Bounding Box Hasil Pemindaian Kamera ──────────────────────────────
    if scan_result:
        # Kotak Wajah
        if scan_result.get("wajah_box") is not None:
            wx, wy, ww, wh = map(int, scan_result["wajah_box"])
            if scan_result.get("wajah_cocok"):
                col_w = HIJAU
                lbl_w = f"Member #{scan_result['wajah_id']} (d={scan_result['wajah_distance']})"
            else:
                col_w = MERAH if scan_result.get("wajah_ada") else ORANYE
                lbl_w = f"Wajah Asing (d={scan_result.get('wajah_distance', '-')})"

            cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), col_w, 2)
            cv2.putText(frame, lbl_w, (wx, max(wy - 8, 85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, col_w, 2)

        # Kotak Plat Nomor
        if scan_result.get("plat_box") is not None and scan_result.get("plat_text"):
            px, py, pw, ph = map(int, scan_result["plat_box"])
            col_p = HIJAU if scan_result.get("plat_valid") else ORANYE
            lbl_p = f"PLAT: {scan_result['plat_text']} ({scan_result.get('plat_conf', 0):.2f})"
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), col_p, 2)
            cv2.putText(frame, lbl_p, (px, max(py - 8, 85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_p, 2)

    # ── 3. Panel Informasi Kiri Layar ────────────────────────────────────────
    pos_y = 105
    if state_sistem == "MEMPROSES":
        w_status = scan_result.get("wajah_cocok") if scan_result else False
        p_status = scan_result.get("plat_valid") if scan_result else False

        txt_w = f"✓ Wajah: Member #{scan_result.get('wajah_id')}" if (scan_result and w_status) else ("⏳ Wajah Tidak Cocok" if (scan_result and scan_result.get("wajah_ada")) else "⏳ Mencari Wajah...")
        txt_p = f"✓ Plat: {scan_result.get('plat_text')}" if (scan_result and p_status) else "⏳ Membaca Plat..."

        cv2.putText(frame, txt_w, (16, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, HIJAU if w_status else ORANYE, 2)
        cv2.putText(frame, txt_p, (16, pos_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, HIJAU if p_status else ORANYE, 2)

    elif state_sistem == "HASIL" and verif_result:
        mem = verif_result.get("member")
        veh = verif_result.get("vehicle")
        if mem:
            cv2.putText(frame, f"Member    : {mem.get('name')}", (16, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, PUTIH, 2)
            pos_y += 24
        if veh:
            cv2.putText(frame, f"Kendaraan : {veh.get('plate_number')}", (16, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, PUTIH, 2)
            pos_y += 24

    # ── 4. Baris Bawah / Status Gerbang ──────────────────────────────────────
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, H - 36), (W, H), HITAM, -1)
    cv2.addWeighted(overlay_bot, 0.75, frame, 0.25, 0, frame)

    info_gate = f"Target: {target_gate} | Izin: {gate_info.get('izin', 'SIAGA')} | Trigger: {gate_info.get('trigger', False)}"
    cv2.putText(frame, info_gate, (16, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, BIRU_CYAN, 1)

    petunjuk = "Web Admin: /parkir/kamera"
    (tw, _), _ = cv2.getTextSize(petunjuk, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(frame, petunjuk, (W - tw - 16, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ABU, 1)


# ─── Program Utama ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ParkVisi - Sistem Kamera AI Terpadu (Masuk & Keluar)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--ip", type=str, default=None, help="URL IP Webcam snapshot")
    parser.add_argument("--laravel", type=str, default="http://127.0.0.1:8000", help="URL server Laravel")
    parser.add_argument("--stream-port", type=int, default=8090, help="Port stream untuk web (default 8090)")
    parser.add_argument("--gate", type=str, default="gate_masuk", choices=["gate_masuk", "gate_keluar"], help="Pintu awal (default: gate_masuk)")
    parser.add_argument("--window", action="store_true", default=False, help="Buka jendela popup OpenCV GUI (default: False/tanpa popup window)")
    parser.add_argument("--no-window", action="store_true", default=False, help="Mode tanpa popup window (default: aktif)")
    args = parser.parse_args()

    buka_gui = args.window and not args.no_window

    print("=" * 65)
    print("  PARKVISI - SISTEM AKSES PARKIR AI (PINTU MASUK & KELUAR)")
    print("=" * 65)
    if buka_gui:
        print("[MODE] Jendela GUI OpenCV aktif.")
    else:
        print("[MODE] Web Streaming aktif (Tanpa popup window).")
        print("       Buka Web Admin di : http://localhost:8000/parkir/kamera")
        print(f"       Siaran MJPEG di   : http://127.0.0.1:{args.stream_port}/stream")

    # 1. Inisialisasi Model LBPH
    print("[1/3] Memuat Model LBPH...")
    if not lbph.load_model():
        print("[WARN] Model LBPH belum ada. Pastikan sudah ada wajah member terdaftar.")
    else:
        print("[INFO] Model LBPH siap digunakan.")

    # 2. Hubungkan ke API Laravel
    print(f"[2/3] Menghubungkan ke Laravel di {args.laravel}...")
    api = LaravelGateAPI(args.laravel)
    cek = api.cek_status_gate()
    if cek.get("error"):
        print(f"[WARN] Gagal terhubung ke Laravel: {cek.get('message')}")
        print("       Pastikan 'php artisan serve --host 0.0.0.0 --port 8000' aktif.")
    else:
        print(f"[INFO] Terhubung ke Laravel API ✓")

    # 3. Buka Kamera
    print("[3/3] Membuka kamera...")
    if args.ip:
        cap = None
        print(f"[INFO] Menggunakan IP Webcam: {args.ip}")
    else:
        cap = buka_kamera(args.cam)
        if cap is None:
            print("[ERROR] Kamera tidak dapat dibuka. Gunakan --cam 1 atau --ip <url>.")
            return

    # Inisialisasi Background Scan Worker
    worker = ScanWorker()
    StreamServer.start(args.stream_port)

    # State Machine: "SIAGA", "MEMPROSES", "HASIL"
    state_sistem = "SIAGA"
    target_gate = args.gate
    waktu_mulai_proses = 0.0
    waktu_mulai_hasil = 0.0
    DURASI_MAKS_PROSES = 9.0   # Batas waktu pencarian wajah & plat (detik)
    DURASI_TAMPIL_HASIL = 5.0  # Durasi tampil banner hasil sebelum kembali siaga

    # Buffer stabilisasi
    history_plat = deque(maxlen=6)
    history_wajah = deque(maxlen=6)

    last_poll_gate = 0.0
    POLL_INTERVAL = 0.4  # Polling trigger dari ESP32 setiap 0.4 detik
    gate_state_info = {"izin": "SIAGA", "trigger": False, "gate": target_gate}

    last_scan_send = 0.0
    SCAN_INTERVAL = 0.8  # Interval pengiriman frame ke worker

    verif_terakhir = None
    scan_terakhir = None

    print("\n[BERJALAN] Sistem otomatis memantau sensor Pintu Masuk & Pintu Keluar.")
    if buka_gui:
        print("           Tekan [Q] di jendela kamera untuk keluar.\n")
    else:
        print("           Tekan Ctrl+C di terminal ini untuk berhenti.\n")

    try:
        while True:
            # ── A. Ambil Frame Kamera ────────────────────────────────────────
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

            # ── B. Polling Status Gate ESP32 dari Laravel ────────────────────
            if sekarang - last_poll_gate >= POLL_INTERVAL:
                last_poll_gate = sekarang
                res_status = api.cek_status_gate()
                if not res_status.get("error"):
                    data_gate = res_status.get("data", {})
                    g_name = data_gate.get("gate", target_gate)
                    g_izin = data_gate.get("izin", "SIAGA")
                    g_trigger = data_gate.get("trigger", False)

                    gate_state_info = {
                        "gate": g_name,
                        "izin": g_izin,
                        "trigger": g_trigger,
                        "status": data_gate.get("status", "SIAGA"),
                    }

                    # Jika ESP32 / tombol web mendeteksi kendaraan
                    if state_sistem == "SIAGA" and (g_trigger or g_izin == "MENUNGGU"):
                        target_gate = g_name
                        nama_pintu = "PINTU KELUAR" if "keluar" in target_gate else "PINTU MASUK"
                        print(f"\n[TRIGGER DITERIMA] Sensor {nama_pintu} ({target_gate}) mendeteksi kendaraan!")
                        state_sistem = "MEMPROSES"
                        waktu_mulai_proses = sekarang
                        history_plat.clear()
                        history_wajah.clear()
                        verif_terakhir = None

                    # Jika gerbang sudah direset kembali ke SIAGA oleh ESP32
                    elif state_sistem == "HASIL" and g_izin == "SIAGA":
                        if sekarang - waktu_mulai_hasil >= 2.0:
                            state_sistem = "SIAGA"
                            verif_terakhir = None
                            history_plat.clear()
                            history_wajah.clear()
                            print(f"[SIAGA] {target_gate} kembali siaga untuk kendaraan berikutnya.\n")

            # ── C. Pemindaian Wajah & Plat Secara Berkala ─────────────────────
            if sekarang - last_scan_send >= SCAN_INTERVAL and not worker.is_busy():
                worker.kirim_frame(frame)
                last_scan_send = sekarang

            hasil_scan = worker.ambil_hasil()
            if hasil_scan:
                scan_terakhir = hasil_scan

            # Jika sedang SIAGA dan kamera melihat plat valid atau wajah yang dikenali (auto-trigger kamera)
            if state_sistem == "SIAGA":
                if (sekarang - waktu_mulai_hasil >= 3.0) and scan_terakhir:
                    if scan_terakhir.get("plat_valid") or scan_terakhir.get("wajah_cocok"):
                        nama_pintu = "PINTU KELUAR" if "keluar" in target_gate else "PINTU MASUK"
                        print(f"\n[DETEKSI KAMERA] {nama_pintu}: Objek terdeteksi di kamera! Memulai pencocokan...")
                        state_sistem = "MEMPROSES"
                        waktu_mulai_proses = sekarang
                        history_plat.clear()
                        history_wajah.clear()
                        verif_terakhir = None

            # ── D. Logika Siklus State Machine ───────────────────────────────
            sisa_waktu = 0.0

            if state_sistem == "MEMPROSES":
                durasi_berjalan = sekarang - waktu_mulai_proses
                sisa_waktu = max(0.0, DURASI_MAKS_PROSES - durasi_berjalan)

                if scan_terakhir:
                    if scan_terakhir.get("plat_valid"):
                        history_plat.append(scan_terakhir["plat_text"])
                    if scan_terakhir.get("wajah_ada"):
                        history_wajah.append((scan_terakhir.get("wajah_id"), scan_terakhir.get("wajah_distance")))

                # Konfirmasi data:
                # 1. Plat nomor terbaca minimal 2x konsisten
                plat_siap = False
                final_plat = None
                if history_plat:
                    c = Counter(history_plat)
                    dom_plat, count_plat = c.most_common(1)[0]
                    if count_plat >= 2:
                        plat_siap = True
                        final_plat = dom_plat

                # 2. Wajah terdeteksi minimal 2 frame
                wajah_siap = len(history_wajah) >= 2
                final_mid = None
                final_dist = None
                if history_wajah:
                    mids = [m for m, d in history_wajah if m is not None]
                    if mids:
                        final_mid = Counter(mids).most_common(1)[0][0]
                    dists = [d for m, d in history_wajah if d is not None]
                    if dists:
                        final_dist = float(np.median(dists))

                timeout = (durasi_berjalan >= DURASI_MAKS_PROSES)

                # Eksekusi verifikasi ke Laravel jika data siap atau waktu habis:
                if (plat_siap and wajah_siap) or timeout:
                    nama_pintu = "PINTU KELUAR" if "keluar" in target_gate else "PINTU MASUK"
                    print(f"[VERIFIKASI {nama_pintu}] Mengirim -> Wajah: {final_mid} | Plat: {final_plat or 'Tidak terbaca'} | Gate: {target_gate}")

                    resp_verif = api.verifikasi_akses(
                        member_id=final_mid,
                        plate_text=final_plat,
                        distance=final_dist,
                        gate=target_gate,
                    )
                    verif_terakhir = resp_verif
                    state_sistem = "HASIL"
                    waktu_mulai_hasil = sekarang

                    st_hasil = resp_verif.get("status", "tolak")
                    ket_hasil = resp_verif.get("keterangan", "")
                    izin_palang = resp_verif.get("izin_gate", "TOLAK")
                    print(f"[HASIL {nama_pintu}] Keputusan: {st_hasil.upper()} | Palang: {izin_palang}")
                    print(f"[KETERANGAN] {ket_hasil}\n")

            elif state_sistem == "HASIL":
                # Otomatis kembali ke SIAGA jika waktu tampilan hasil selesai
                if sekarang - waktu_mulai_hasil >= DURASI_TAMPIL_HASIL:
                    state_sistem = "SIAGA"
                    verif_terakhir = None
                    history_plat.clear()
                    history_wajah.clear()
                    print(f"[SIAGA] Sistem siap memantau kendaraan berikutnya.\n")

            # ── E. Render UI Tampilan Kamera (selalu dijalankan tiap frame) ───
            gambar_interface(
                frame=frame,
                state_sistem=state_sistem,
                target_gate=target_gate,
                gate_info=gate_state_info,
                scan_result=scan_terakhir,
                verif_result=verif_terakhir,
                sisa_waktu=sisa_waktu,
                interval_detik=SCAN_INTERVAL,
            )

            # Kirim frame dengan overlay deteksi ke stream web
            StreamServer.update(frame)

            # Jika mode headless (default), lewati cv2.imshow agar tidak memunculkan window popup
            if not buka_gui:
                time.sleep(0.01)
                continue

            cv2.imshow("ParkVisi - Sistem Akses Parkir Terpadu", frame)
            # ── F. Kontrol Keyboard (Hanya jika GUI aktif) ───────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("1"), ord("m"), ord("M"), ord(" ")):
                # Trigger Manual Pintu Masuk
                target_gate = "gate_masuk"
                print("\n[MANUAL] Trigger PINTU MASUK (gate_masuk)...")
                api.trigger_manual("gate_masuk")
                state_sistem = "MEMPROSES"
                waktu_mulai_proses = sekarang
                history_plat.clear()
                history_wajah.clear()
                verif_terakhir = None
            elif key in (ord("2"), ord("k"), ord("K")):
                # Trigger Manual Pintu Keluar
                target_gate = "gate_keluar"
                print("\n[MANUAL] Trigger PINTU KELUAR (gate_keluar)...")
                api.trigger_manual("gate_keluar")
                state_sistem = "MEMPROSES"
                waktu_mulai_proses = sekarang
                history_plat.clear()
                history_wajah.clear()
                verif_terakhir = None
            elif key in (ord("r"), ord("R")):
                # Reset Manual Gerbang
                print(f"[MANUAL] Mereset gerbang ({target_gate}) ke SIAGA...")
                api.reset_gate(target_gate)
                state_sistem = "SIAGA"
                verif_terakhir = None
                history_plat.clear()
                history_wajah.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Menghentikan sistem (Ctrl+C)...")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("[SELESAI] Program berhenti.")


if __name__ == "__main__":
    main()
