"""Alur lengkap baca plat: lokasi -> preprocessing -> OCR -> evaluasi format/confidence.

Uji tanpa server (foto di data/plates, gambar tiap langkah disimpan di data/debug):
    python -m app.vision.plate_reader
"""
import cv2

from app.config import DEBUG_DIR, PLATE_TEST_DIR
from app.intelligence import plate_rules
from app.vision import plate_detector, plate_ocr, plate_preprocess


def read_plate(img, debug=None):
    """Return dict: success, plate_text, plate_confidence, plate_valid, plate_candidates, box, message."""
    lokasi = plate_detector.find_plate_candidates(img, debug)
    if not lokasi:
        return {"success": False, "plate_text": None, "plate_confidence": None, "plate_valid": False,
                "plate_candidates": [], "box": None, "message": "Lokasi plat tidak ditemukan."}

    hasil_ocr = []
    for i, k in enumerate(lokasi):
        varian = plate_preprocess.prepare_variants(k["crop"], debug if i == 0 else None)
        for r in plate_ocr.read_variants(varian):
            r["box"] = list(map(int, k["box"]))
            hasil_ocr.append(r)

    kandidat = plate_rules.evaluasi(hasil_ocr)
    if not kandidat:
        return {"success": False, "plate_text": None, "plate_confidence": None, "plate_valid": False,
                "plate_candidates": [], "box": None, "message": "Plat ditemukan tetapi karakter tidak terbaca."}

    terbaik = kandidat[0]
    return {
        "success": terbaik["valid"],
        "plate_text": terbaik["text"],
        "plate_confidence": terbaik["confidence"],
        "plate_valid": terbaik["valid"],
        "plate_candidates": [{"text": k["text"], "confidence": k["confidence"], "valid": k["valid"]} for k in kandidat],
        "box": terbaik["box"],
        "message": "OK" if terbaik["valid"] else "Hasil OCR tidak memenuhi format atau confidence minimum.",
    }


def save_debug(debug, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for nama, img in debug.items():
        cv2.imwrite(str(folder / f"{nama}.png"), img)


def main():
    PLATE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    berkas = sorted(p for p in PLATE_TEST_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not berkas:
        print(f"Tidak ada foto di {PLATE_TEST_DIR}. Taruh foto uji plat di folder itu.")
        return
    for p in berkas:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: gagal dibaca")
            continue
        debug = {}
        hasil = read_plate(img, debug)
        save_debug(debug, DEBUG_DIR / p.stem)
        print(f"{p.name}: {hasil['plate_text']} (conf {hasil['plate_confidence']}, valid {hasil['plate_valid']}) - {hasil['message']}")
        for k in hasil["plate_candidates"]:
            print(f"    kandidat: {k}")


if __name__ == "__main__":
    main()
