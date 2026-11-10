import urllib.request
from datetime import datetime

from flask import Blueprint, jsonify, request

from app.config import (DEBUG_DIR, MEMBER_MAP, PHONE_CAMERA_URL,
                        YOLO_ACCEPT_CONF)
from app.vision import plate_reader, yolo_recognizer
from app.vision.face_preprocess import decode_image

bp = Blueprint("api", __name__)


# ---------------------------------------------------------------------------
# WAJAH (YOLO)
# ---------------------------------------------------------------------------
@bp.post("/detect")
def detect():
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="Foto tidak dikirim."), 400
    img = decode_image(f.read())
    if img is None:
        return jsonify(face=False)
    return jsonify(face=yolo_recognizer.predict(img) is not None)


@bp.post("/recognize")
def recognize():
    """Kenali wajah dengan YOLO.

    Response JSON (format sama seperti sebelumnya):
      face_detected, matched, member_id, distance, warning
    Tambahan: confidence, predicted_name
    """
    file = request.files.get("photo")
    if file is None:
        return jsonify(error="Foto tidak ditemukan."), 422

    img = decode_image(file.read())
    if img is None:
        return jsonify(error="File bukan gambar yang valid."), 422

    result = yolo_recognizer.predict(img)

    if result is None:
        return jsonify(
            face_detected=False, matched=False, member_id=None, distance=None,
            warning="Wajah tidak terdeteksi dalam frame. "
                    "Pastikan wajah menghadap kamera dan cukup terang."
        ), 200

    nama, conf = result
    member_id = MEMBER_MAP.get(nama)
    matched = member_id is not None and conf >= YOLO_ACCEPT_CONF

    return jsonify(
        face_detected=True,
        matched=matched,
        member_id=member_id if matched else None,
        distance=round(1 - conf, 3),
        confidence=round(conf, 3),
        predicted_name=nama,
        warning=None if matched else (
            "Wajah terdeteksi tetapi tidak cocok dengan member manapun (orang asing / belum terdaftar)."
        ),
    )


# ---------------------------------------------------------------------------
# STUB: endpoint lama LBPH (hanya kalau Laravel masih memanggilnya)
# Hapus blok ini kalau sudah dipastikan tidak ada pemanggil.
# ---------------------------------------------------------------------------
@bp.post("/enroll")
def enroll():
    return jsonify(message="Pendaftaran wajah via web dinonaktifkan. Member dilatih lewat YOLO."), 410


@bp.post("/retrain")
def retrain():
    return jsonify(message="Tidak ada retrain di server. Model YOLO dilatih di luar."), 410


@bp.delete("/members/<int:member_id>")
def remove_member(member_id):
    return jsonify(message="Tidak ada data member di AI service."), 410


@bp.get("/health")
def health():
    return jsonify(status="ok")


# ---------------------------------------------------------------------------
# PLAT NOMOR (tidak diubah)
# ---------------------------------------------------------------------------
def _baca_plat(img):
    """Jalankan alur plat; tambahkan ?debug=1 untuk menyimpan gambar tiap langkah."""
    debug = {} if request.args.get("debug") else None
    try:
        hasil = plate_reader.read_plate(img, debug)
    except RuntimeError as e:
        return jsonify(success=False, message=str(e)), 503
    if debug:
        folder = DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_reader.save_debug(debug, folder)
        hasil["debug_dir"] = str(folder)
    return jsonify(hasil), (200 if hasil["success"] else 422)


@bp.post("/plate/read")
def plate_read():
    f = request.files.get("photo")
    if f is None:
        return jsonify(success=False, message="Foto tidak ditemukan."), 422
    img = decode_image(f.read())
    if img is None:
        return jsonify(success=False, message="File bukan gambar yang valid."), 422
    return _baca_plat(img)


@bp.get("/detect-phone-plate")
def detect_phone_plate():
    """Dipanggil Laravel (verifyAccess): ambil satu frame dari kamera HP lalu baca plat."""
    try:
        with urllib.request.urlopen(PHONE_CAMERA_URL, timeout=5) as r:
            data = r.read()
    except Exception as e:
        return jsonify(success=False, message=f"Kamera HP tidak dapat dihubungi: {e}"), 503
    img = decode_image(data)
    if img is None:
        return jsonify(success=False, message="Frame kamera HP tidak valid."), 422
    return _baca_plat(img)