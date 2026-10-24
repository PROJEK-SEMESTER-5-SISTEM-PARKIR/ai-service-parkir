import cv2
from app.config import MIN_FACE_PX

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(gray):
    """Kembalikan (x, y, w, h) wajah terbesar, atau None."""
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])