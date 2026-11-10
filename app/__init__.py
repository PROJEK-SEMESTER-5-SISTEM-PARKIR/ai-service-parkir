from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException


def create_app():
    app = Flask(__name__)

    # Muat model YOLO saat service menyala, supaya request pertama tidak lambat
    from app.vision import yolo_recognizer
    try:
        yolo_recognizer.load()
        app.logger.info("Model YOLO siap digunakan.")
    except Exception:
        app.logger.exception("Model YOLO gagal dimuat. Cek YOLO_MODEL_PATH di config.py dan pastikan models/best.pt ada.")
        raise

    from app.api.routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(Exception)
    def on_error(e):
        # Semua error dibalas JSON berisi "error", supaya Laravel bisa menampilkan penyebab aslinya
        if isinstance(e, HTTPException):
            return jsonify(error=e.description), e.code
        app.logger.exception("Unhandled error")
        return jsonify(error=f"Kesalahan di Python service: {e}"), 500

    return app