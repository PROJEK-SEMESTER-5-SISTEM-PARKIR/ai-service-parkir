"""Alur lengkap baca plat: lokasi -> preprocessing -> OCR -> evaluasi format/confidence.

Uji tanpa server (foto di data/plates, gambar tiap langkah disimpan di data/debug):
    python -m app.vision.plate_reader
"""
import cv2

from app.config import DEBUG_DIR, PLATE_EARLY_STOP, PLATE_TEST_DIR
from app.intelligence import plate_rules
from app.vision import plate_detector, plate_ocr, plate_preprocess


def _gagal(pesan):
    return {"success": False, "plate_text": None, "plate_confidence": None, "plate_valid": False,
            "plate_candidates": [], "box": None, "message": pesan}


def read_plate(img, debug=None, fast=False):
    """Return dict: success, plate_text, plate_confidence, plate_valid, plate_candidates, box, message.

    fast=True: mode real-time (resize ke 640px, OCR 1 PSM x 3 varian).
               Lebih cepat ~8-10x, cukup untuk sistem parkir live.
    fast=False: mode akurasi tinggi (4 PSM x 8+ varian) untuk foto statis.
    """
    # Mode cepat: perkecil frame agar detektor tidak bekerja pada resolusi penuh
    scale = 1.0
    if fast:
        h, w = img.shape[:2]
        if w > 640:
            scale = 640.0 / w
            img = cv2.resize(img, (640, int(h * scale)))

    lokasi = plate_detector.find_plate_candidates(img, debug)
    if not lokasi:
        return _gagal("Lokasi plat tidak ditemukan.")

    semua = []
    for i, k in enumerate(lokasi):
        varian = plate_preprocess.prepare_variants(k["crop"], debug if i == 0 else None, fast=fast)
        hasil_k = []
        for r in plate_ocr.read_variants(varian, fast=fast):
            # Kembalikan koordinat box ke resolusi asli
            bx, by, bw, bh = k["box"]
            if scale != 1.0:
                bx = int(bx / scale); by = int(by / scale)
                bw = int(bw / scale); bh = int(bh / scale)
            r["box"] = [bx, by, bw, bh]
            hasil_k.append(r)

        evaluasi_k = plate_rules.evaluasi(hasil_k)
        semua.extend(evaluasi_k)
        if evaluasi_k and evaluasi_k[0]["valid"] and evaluasi_k[0]["confidence"] >= PLATE_EARLY_STOP:
            break

    unik = {}
    for item in semua:
        lama = unik.get(item["text"])
        if lama is None or item["confidence"] > lama["confidence"]:
            unik[item["text"]] = item
    kandidat = sorted(unik.values(), key=plate_rules.kunci_urut)
    if not kandidat:
        return _gagal("Plat ditemukan tetapi karakter tidak terbaca.")

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