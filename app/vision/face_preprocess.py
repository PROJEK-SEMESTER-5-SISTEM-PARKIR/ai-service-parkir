import cv2
import numpy as np
from app.config import FACE_SIZE

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def decode_image(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def crop_face(gray, box, margin=0.1):
    x, y, w, h = box
    m = int(margin * w)
    y0, y1 = max(0, y - m), min(gray.shape[0], y + h + m)
    x0, x1 = max(0, x - m), min(gray.shape[1], x + w + m)
    return gray[y0:y1, x0:x1]

def normalize(face):
    """Samakan ukuran dan ratakan pencahayaan."""
    return _clahe.apply(cv2.resize(face, FACE_SIZE))

def _rotate(face, angle):
    h, w = face.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(face, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

def augment(face):
    """Perbanyak sampel training dari satu foto."""
    return [
        face,
        cv2.flip(face, 1),
        cv2.convertScaleAbs(face, alpha=1.0, beta=25),
        cv2.convertScaleAbs(face, alpha=1.0, beta=-25),
        _rotate(face, 5),
        _rotate(face, -5),
        cv2.GaussianBlur(face, (3, 3), 0),
    ]