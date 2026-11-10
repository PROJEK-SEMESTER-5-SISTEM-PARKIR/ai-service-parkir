"""
Pengenalan wajah dengan YOLO (menggantikan LBPH).

Taruh file ini di: app/vision/yolo_recognizer.py
"""
from ultralytics import YOLO

from app.config import YOLO_MIN_CONF, YOLO_MODEL_PATH

_model = None


def load():
    """Muat model sekali saja (bukan tiap request)."""
    global _model
    if _model is None:
        _model = YOLO(str(YOLO_MODEL_PATH))
    return _model


def predict_detail(img_bgr):
    """Return None kalau tidak ada wajah.

    Kalau ada, return dict untuk wajah TERBESAR (paling dekat kamera):
      {"name": str, "conf": float, "box": (x, y, w, h)}
    """
    r = load().predict(img_bgr, conf=YOLO_MIN_CONF, verbose=False)[0]
    if len(r.boxes) == 0:
        return None

    def luas(b):
        x1, y1, x2, y2 = b.xyxy[0]
        return float((x2 - x1) * (y2 - y1))

    b = max(r.boxes, key=luas)
    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
    return {
        "name": r.names[int(b.cls)],
        "conf": float(b.conf),
        "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),  # format x, y, w, h
    }


def predict(img_bgr):
    """Dipakai routes.py: (nama_kelas, confidence) atau None."""
    d = predict_detail(img_bgr)
    return None if d is None else (d["name"], d["conf"])