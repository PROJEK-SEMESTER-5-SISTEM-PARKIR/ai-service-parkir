# untuk run hanya file ini untuk mengetes pengenalan wajah, tanpa harus menjalankan server Flask.
# python -m app.vision.recognizer
import cv2

from app.config import UNKNOWN_THRESHOLD
from app.intelligence import lbph
from app.vision import face_detector, face_preprocess


def recognize_frame(frame):
    """
    Mendeteksi wajah pada satu frame kamera,
    lalu mencocokkannya dengan model LBPH.

    Return:
        face_detected (bool), box (x,y,w,h)|None, member_id|None, distance|None, warning|None
    """
    gray = face_preprocess.to_gray(frame)
    box = face_detector.detect_face(gray)

    # ── Tidak ada wajah di frame ────────────────────────────────────────────
    if box is None:
        return False, None, None, None, "Wajah tidak terdeteksi. Hadapkan wajah ke kamera."

    # ── Ada wajah → cocokkan ─────────────────────────────────────────────────
    face = face_preprocess.normalize(
        face_preprocess.crop_face(gray, box)
    )

    result = lbph.predict(face)

    if result is None:
        return True, box, None, None, "Model belum tersedia (belum ada yang didaftarkan)."

    member_id, distance = result

    if distance > UNKNOWN_THRESHOLD:
        return True, box, None, distance, (
            f"Wajah terdeteksi (distance={distance:.1f}) "
            "tapi tidak cocok dengan member manapun (orang asing)."
        )

    return True, box, member_id, distance, None


def main():
    # Muat model LBPH yang sudah tersedia.
    if not lbph.load_model():
        print("Model LBPH tidak ditemukan.")
        print("Daftarkan wajah terlebih dahulu melalui endpoint /enroll.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Kamera tidak dapat dibuka.")
        print("Periksa apakah kamera sedang digunakan aplikasi lain.")
        return

    print("Kamera aktif. Tekan Q untuk keluar.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Gagal membaca frame kamera.")
                break

            face_detected, box, member_id, distance, warning = recognize_frame(frame)

            if not face_detected:
                # ── Warning: tidak ada wajah ────────────────────────────────
                cv2.putText(
                    frame,
                    "⚠ Wajah tidak terdeteksi!",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),   # oranye
                    2,
                )

            elif box is not None:
                x, y, w, h = map(int, box)

                if member_id is not None:
                    # ── Dikenali ────────────────────────────────────────────
                    label = f"Member ID: {member_id}"
                    if distance is not None:
                        label += f" | Distance: {distance:.1f}"
                    color = (0, 200, 0)     # hijau
                else:
                    # ── Wajah ada tapi tidak cocok / asing ──────────────────
                    label = warning or "Wajah tidak dikenal"
                    if distance is not None:
                        label = f"Asing | Distance: {distance:.1f}"
                    color = (0, 0, 255)     # merah

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2,
                )

            cv2.imshow("ParkVisi - Pengenalan Wajah LBPH", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
