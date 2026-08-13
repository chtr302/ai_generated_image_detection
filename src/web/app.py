from __future__ import annotations

import gc
import logging
import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    from .services.inference import (
        InputRef,
        analyze_upload,
        clear_model_cache_and_worker,
        data_url_for_image,
        fetch_url_image,
        model_detail_payload,
        validate_image_name,
        validate_url,
    )
except ImportError:
    from services.inference import (
        InputRef,
        analyze_upload,
        clear_model_cache_and_worker,
        data_url_for_image,
        fetch_url_image,
        model_detail_payload,
        validate_image_name,
        validate_url,
    )


BASE_DIR = Path(__file__).resolve().parent
MODEL_ENABLED_DEFAULT = os.getenv("AIGID_MODEL_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
MODEL_STATE = {"enabled": MODEL_ENABLED_DEFAULT}
SHOW_ACCESS_LOGS = os.getenv("AIGID_ACCESS_LOGS", "0").strip().lower() in {"1", "true", "on", "yes"}
MODEL_CONTROL_KEY_ENV = "AIGID_MODEL_CONTROL_KEY"
MODEL_CONTROL_KEY_HEADER = "X-Model-Control-Key"


def configure_logging() -> None:
    if not SHOW_ACCESS_LOGS:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)


def unload_model_memory() -> None:
    # chuthich: Tat model bang cach dung worker process dang giu ONNX session trong RAM.
    clear_model_cache_and_worker()
    gc.collect()


def model_state_payload() -> dict[str, object]:
    return {
        "enabled": MODEL_STATE["enabled"],
        "mode": "on" if MODEL_STATE["enabled"] else "off",
        "message": "Model dang bat." if MODEL_STATE["enabled"] else "Model dang tat. Chuc nang phan tich da bi khoa.",
    }


def configured_control_key() -> str:
    return os.getenv(MODEL_CONTROL_KEY_ENV, "").strip()


def submitted_control_key() -> str:
    payload = request.get_json(silent=True) or {}
    return (
        request.headers.get(MODEL_CONTROL_KEY_HEADER)
        or request.form.get("key")
        or payload.get("key")
        or ""
    ).strip()


def model_control_key_status_payload() -> dict[str, object]:
    return {
        "configured": bool(configured_control_key()),
        "required": bool(configured_control_key()),
        "env": MODEL_CONTROL_KEY_ENV,
        "header": MODEL_CONTROL_KEY_HEADER,
    }


def control_auth_error():
    expected_key = configured_control_key()
    if not expected_key:
        return None

    submitted_key = submitted_control_key()
    if not submitted_key:
        return (
            jsonify(
                {
                    "error": "Thieu model control key.",
                    "error_code": "missing_model_control_key",
                    **model_control_key_status_payload(),
                }
            ),
            401,
        )

    if not secrets.compare_digest(submitted_key, expected_key):
        return (
            jsonify(
                {
                    "error": "Model control key khong hop le.",
                    "error_code": "invalid_model_control_key",
                    **model_control_key_status_payload(),
                }
            ),
            403,
        )
    return None


def model_disabled_response():
    payload = model_state_payload()
    payload["error"] = "Model dang tat. Chuc nang phan tich hien dang tam khoa."
    return jsonify(payload), 503


def create_app() -> Flask:
    configure_logging()

    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "Tep qua lon. Gioi han request la 64 MB."}), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "ai-generated-image-detection-web"})

    @app.get("/api/model-state")
    def model_state():
        return jsonify(model_state_payload())

    @app.get("/api/model-detail")
    def model_detail():
        return jsonify(model_detail_payload(bool(MODEL_STATE["enabled"])))

    @app.post("/api/model-control-key/test")
    def test_model_control_key():
        auth_error = control_auth_error()
        if auth_error is not None:
            return auth_error

        configured = bool(configured_control_key())
        return jsonify(
            {
                "valid": True,
                "message": "Model control key hop le." if configured else "Chua cau hinh model control key; lenh dieu khien dang mo.",
                **model_control_key_status_payload(),
            }
        )

    @app.post("/api/model-state")
    def set_model_state():
        auth_error = control_auth_error()
        if auth_error is not None:
            return auth_error

        payload = request.get_json(silent=True) or {}
        if "enabled" not in payload:
            return jsonify({"error": "Thieu truong enabled.", "error_code": "missing_enabled"}), 400
        if not isinstance(payload["enabled"], bool):
            return jsonify({"error": "Truong enabled phai la boolean true/false.", "error_code": "invalid_enabled"}), 400

        enabled = payload["enabled"]
        MODEL_STATE["enabled"] = enabled
        if not enabled:
            unload_model_memory()
        return jsonify(model_state_payload())

    @app.post("/api/load-url")
    def load_url():
        if not MODEL_STATE["enabled"]:
            return model_disabled_response()

        url = request.form.get("url", "").strip()
        ok, error = validate_url(url)
        if not ok:
            return jsonify({"error": error}), 400

        data, error = fetch_url_image(url)
        if error or data is None:
            return jsonify({"error": error}), 400

        return jsonify(
            {
                "input_name": url,
                "input_kind": "url",
                "preview_data_url": data_url_for_image(url, data),
                "size_bytes": len(data),
            }
        )

    @app.post("/api/analyze")
    def analyze():
        if not MODEL_STATE["enabled"]:
            return model_disabled_response()

        uploaded = request.files.get("image")
        url = request.form.get("url", "").strip()

        if uploaded and uploaded.filename:
            data = uploaded.read()
            original_name = uploaded.filename
            safe_name = secure_filename(original_name) or "uploaded_image"
            ok, error = validate_image_name(original_name, len(data))
            if not ok:
                return jsonify({"error": error}), 400

            input_ref = InputRef(name=safe_name, kind="upload")
            try:
                return jsonify(analyze_upload(input_ref, data))
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 500

        if url:
            ok, error = validate_url(url)
            if not ok:
                return jsonify({"error": error}), 400

            data, error = fetch_url_image(url)
            if error or data is None:
                return jsonify({"error": error}), 400

            input_ref = InputRef(name=url, kind="url")
            try:
                result = analyze_upload(input_ref, data)
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 500

            result["preview_data_url"] = data_url_for_image(url, data)
            return jsonify(result)

        return jsonify({"error": "Vui long upload anh hoac nhap URL anh."}), 400

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
