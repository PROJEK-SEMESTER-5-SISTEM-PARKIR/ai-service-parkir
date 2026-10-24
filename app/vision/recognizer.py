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
        box, member_id, distance
    """
    gray = face_preprocess.to_gray(frame)
    box = face_detector.detect_face(gray)

    if box is None:
        return None, None, None

    face = face_preprocess.normalize(
        face_preprocess.crop_face(gray, box)
    )

    result = lbph.predict(face)

    if result is None:
        return box, None, None

    member_id, distance = result

    if distance > UNKNOWN_THRESHOLD:
        return box, None, distance

    return box, member_id, distance


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

            box, member_id, distance = recognize_frame(frame)

            if box is not None:
                # Box diasumsikan berbentuk (x, y, w, h).
                x, y, w, h = map(int, box)

                if member_id is not None:
                    label = f"Member ID: {member_id}"
                    if distance is not None:
                        label += f" | Distance: {distance:.1f}"

                    color = (0, 200, 0)
                elif distance is not None:
                    label = "Wajah tidak dikenal"
                    color = (0, 0, 255)
                else:
                    label = "Model belum tersedia"
                    color = (0, 165, 255)

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Tidak ada wajah terdeteksi",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
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
