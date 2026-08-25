from __future__ import annotations

import base64
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_LIMIT_BYTES = 10 * 1024 * 1024

MODEL_DIR_ENV = "AIGID_MODEL_DIR"
AI_CLASS_INDEX_ENV = "AIGID_AI_CLASS_INDEX"
WORKER_TIMEOUT_ENV = "AIGID_WORKER_TIMEOUT_SECONDS"

SUPPORTED_IMAGES = {"jpg", "jpeg", "png", "webp"}
IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
MODEL_SPECS = (
    {
        "key": "ai_detection",
        "name": "AI Detection",
        "filename": "AI Detection.onnx",
        "image_size": 256,
        "output_kind": "logits",
    },
    {
        "key": "ai_detection_physic",
        "name": "AI Detection + Physic",
        "filename": "AI Detection + Physic.onnx",
        "image_size": 256,
        "output_kind": "prob_ai",
    },
)

_SESSIONS: dict[str, Any] = {}
_WORKER_PROCESS: mp.Process | None = None
_WORKER_CONN: Any | None = None
_WORKER_LOCK = Lock()


@dataclass(frozen=True)
class InputRef:
    name: str
    kind: str


def extension_of(name: str) -> str:
    clean = name.split("?", 1)[0].split("#", 1)[0]
    return clean.rsplit(".", 1)[-1].lower() if "." in clean else ""


def safe_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def safe_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def model_dir() -> Path:
    raw_path = os.getenv(MODEL_DIR_ENV, "").strip()
    return Path(raw_path).expanduser() if raw_path else DEFAULT_MODEL_DIR


def model_path(spec: dict[str, Any]) -> Path:
    return model_dir() / str(spec["filename"])


def available_model_payloads() -> list[dict[str, Any]]:
    return [
        {
            "key": spec["key"],
            "name": spec["name"],
            "filename": spec["filename"],
            "image_size": spec["image_size"],
            "path": str(model_path(spec)),
            "found": model_path(spec).exists(),
        }
        for spec in MODEL_SPECS
    ]


def model_detail_payload(model_enabled: bool) -> dict[str, Any]:
    models = available_model_payloads()
    ready = all(item["found"] for item in models)
    return {
        "name": "AI Detection ONNX Ensemble (2 models)",
        "engine": "onnxruntime",
        "mode": "Sẵn sàng" if ready else "Thiếu file model",
        "model_enabled": model_enabled,
        "model_dir_env": MODEL_DIR_ENV,
        "model_dir": str(model_dir()),
        "models_ready": ready,
        "models": models,
        "ai_class_index": safe_env_int(AI_CLASS_INDEX_ENV, 1),
        "ai_class_index_env": AI_CLASS_INDEX_ENV,
        "worker_running": model_worker_running(),
        "worker_pid": model_worker_pid(),
        "max_image_mb": IMAGE_LIMIT_BYTES // (1024 * 1024),
        "supported_images": sorted(SUPPORTED_IMAGES),
    }


def validate_image_name(name: str, size: int | None = None) -> tuple[bool, str | None]:
    ext = extension_of(name)
    if ext not in SUPPORTED_IMAGES:
        return False, "Định dạng không hỗ trợ. Chỉ nhận jpg, jpeg, png hoặc webp."
    if size is not None and size > IMAGE_LIMIT_BYTES:
        return False, "Ảnh quá lớn. Giới hạn là 10 MB."
    return True, None


def validate_url(url: str) -> tuple[bool, str | None]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "URL phải dùng http hoặc https."
    if not parsed.netloc:
        return False, "URL thiếu tên miền hoặc host."
    return True, None


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def data_url_for_image(name: str, data: bytes) -> str:
    mime_type = detect_image_mime(data) or IMAGE_MIME_TYPES.get(extension_of(name), "image/png")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def fetch_url_image(url: str) -> tuple[bytes | None, str | None]:
    ok, error = validate_url(url)
    if not ok:
        return None, error

    request = Request(url, headers={"User-Agent": "AI-Image-Detection-Web/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > IMAGE_LIMIT_BYTES:
                return None, "Ảnh URL quá lớn. Giới hạn là 10 MB."

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type and content_type not in IMAGE_MIME_TYPES.values():
                return None, "URL không trả về content-type ảnh hợp lệ."

            data = response.read(IMAGE_LIMIT_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 403:
            return None, "URL từ chối truy cập ảnh. Vui lòng dùng link ảnh công khai khác."
        if exc.code == 404:
            return None, "Không tìm thấy ảnh tại URL này. Vui lòng kiểm tra lại đường dẫn."
        return None, "Không tải được ảnh từ URL này. Vui lòng kiểm tra lại đường dẫn ảnh."
    except URLError:
        return None, "Không kết nối được tới URL này. Vui lòng kiểm tra tên miền hoặc kết nối mạng."
    except TimeoutError:
        return None, "Tải ảnh từ URL quá lâu. Vui lòng thử lại hoặc dùng URL khác."
    except Exception:
        return None, "Không tải được ảnh từ URL này. Vui lòng kiểm tra lại đường dẫn ảnh."

    if len(data) > IMAGE_LIMIT_BYTES:
        return None, "Ảnh URL quá lớn. Giới hạn là 10 MB."
    if not data:
        return None, "URL không trả về dữ liệu ảnh."
    if detect_image_mime(data) is None and extension_of(urlparse(url).path) not in SUPPORTED_IMAGES:
        return None, "URL không trả về dữ liệu ảnh hợp lệ."
    return data, None


def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError("Tệp không phải ảnh hợp lệ.") from exc
    return image.convert("RGB")


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    resize_size = image_size + 32
    width, height = image.size
    scale = resize_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)

    left = max(0, (resized.width - image_size) // 2)
    top = max(0, (resized.height - image_size) // 2)
    cropped = resized.crop((left, top, left + image_size, top + image_size))

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(array.astype(np.float32), axis=0)


def get_session(spec: dict[str, Any]) -> Any:
    key = str(spec["key"])
    if key in _SESSIONS:
        return _SESSIONS[key]

    path = model_path(spec)
    if not path.exists():
        raise RuntimeError(f"Không tìm thấy model: {path}")

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("Chưa cài onnxruntime. Hãy cài bằng: pip install onnxruntime") from exc

    _SESSIONS[key] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def clear_model_cache() -> None:
    _SESSIONS.clear()


def model_worker_running() -> bool:
    return _WORKER_PROCESS is not None and _WORKER_PROCESS.is_alive()


def model_worker_pid() -> int | None:
    return _WORKER_PROCESS.pid if model_worker_running() else None


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def probabilities_from_output(
    spec: dict[str, Any],
    output_values: np.ndarray,
    ai_class_index: int,
) -> tuple[np.ndarray, int, float, float, float]:
    if output_values.ndim == 2:
        output_values = output_values[0]
    if output_values.ndim == 0:
        output_values = output_values.reshape(1)
    if output_values.ndim != 1:
        raise RuntimeError(f"Output model {spec['name']} không đúng dạng vector.")

    output_kind = str(spec.get("output_kind", "probabilities"))
    if output_values.size == 1:
        if output_kind == "logit_ai":
            ai_probability = float(1.0 / (1.0 + np.exp(-output_values[0])))
        else:
            ai_probability = float(output_values[0])
        ai_probability = float(np.clip(ai_probability, 0.0, 1.0))
        real_probability = 1.0 - ai_probability
        probabilities = np.asarray([real_probability, ai_probability], dtype=np.float32)
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        return probabilities, predicted_index, real_probability, ai_probability, confidence

    if ai_class_index < 0 or ai_class_index >= output_values.size:
        raise RuntimeError(f"{AI_CLASS_INDEX_ENV} không hợp lệ với output {output_values.size} lớp.")

    probabilities = softmax(output_values) if output_kind == "logits" else output_values
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    predicted_index = int(np.argmax(probabilities))
    ai_probability = float(probabilities[ai_class_index])
    real_index = 0 if ai_class_index != 0 else 1
    real_probability = float(probabilities[real_index])
    confidence = float(probabilities[predicted_index])
    return probabilities, predicted_index, real_probability, ai_probability, confidence


def run_model(spec: dict[str, Any], image: Image.Image, ai_class_index: int) -> dict[str, Any]:
    session = get_session(spec)
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    tensor = preprocess_image(image, int(spec["image_size"]))
    raw_output = session.run([model_output.name], {model_input.name: tensor})[0]
    output_values = np.asarray(raw_output, dtype=np.float32)
    probabilities, predicted_index, real_probability, ai_probability, confidence = probabilities_from_output(
        spec,
        output_values,
        ai_class_index,
    )
    vote_label = "AI-generated" if predicted_index == ai_class_index else "Real"
    return {
        "key": spec["key"],
        "model": spec["name"],
        "output_name": model_output.name,
        "raw_output": [round(float(value), 6) for value in np.ravel(output_values).tolist()],
        "probabilities": [round(float(value), 6) for value in probabilities.tolist()],
        "prob_real": round(real_probability, 6),
        "prob_ai": round(ai_probability, 6),
        "ai_score": round(ai_probability, 4),
        "predicted_index": predicted_index,
        "confidence": round(confidence, 6),
        "vote": vote_label,
    }


def vote(scores: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(scores, key=lambda item: float(item["confidence"]))
    final_label = selected["vote"]

    return {
        "final_label": final_label,
        "final_score": selected["ai_score"],
        "selected_confidence": selected["confidence"],
        "selected_model": {
            "key": selected["key"],
            "name": selected["model"],
            "vote": selected["vote"],
            "confidence": selected["confidence"],
            "prob_real": selected["prob_real"],
            "prob_ai": selected["prob_ai"],
            "raw_output": selected["raw_output"],
        },
        "decision_status": "selected_highest_confidence",
        "low_confidence": False,
    }


def build_explanation(scores: list[dict[str, Any]], voted: dict[str, Any]) -> tuple[str, str | None]:
    names = ", ".join(
        f"{item['model']} raw={item['raw_output']} vote={item['vote']} confidence={item['confidence']}"
        for item in scores
    )
    selected = voted["selected_model"]["name"]
    label = voted["final_label"]
    text = f"Kết quả chọn theo model có confidence cao nhất: {selected} => {label}. Output từng model: {names}."
    return text, None


def analyze_upload_local(input_ref: InputRef, data: bytes) -> dict[str, Any]:
    # chuthich: Ham nay chi chay trong worker process, noi model ONNX duoc load vao RAM.
    started = time.perf_counter()
    image = load_image(data)
    ai_class_index = safe_env_int(AI_CLASS_INDEX_ENV, 1)
    scores = [run_model(spec, image, ai_class_index) for spec in MODEL_SPECS]
    voted = vote(scores)
    text, warning = build_explanation(scores, voted)

    return {
        "job_id": f"job_{int(time.time() * 1000):x}",
        "input_name": input_ref.name,
        "input_kind": input_ref.kind,
        **voted,
        "model_scores": scores,
        "explanation": {
            "type": "onnx_ensemble_scores",
            "concepts": [],
            "text": text,
        },
        "warning": warning,
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "engine": "onnxruntime",
    }


def worker_loop(conn: Any) -> None:
    # chuthich: Worker giu ONNX session rieng; tat model se kill process nay de tra RAM cho OS.
    try:
        while True:
            try:
                message = conn.recv()
            except (KeyboardInterrupt, EOFError, OSError):
                break

            command = message.get("command")
            if command == "stop":
                break
            if command != "analyze":
                conn.send({"ok": False, "error": "Lệnh worker không hợp lệ."})
                continue

            try:
                input_ref = InputRef(**message["input_ref"])
                result = analyze_upload_local(input_ref, message["data"])
                conn.send({"ok": True, "result": result})
            except Exception as exc:
                conn.send({"ok": False, "error": str(exc)})
    except KeyboardInterrupt:
        pass
    finally:
        clear_model_cache()
        conn.close()


def start_model_worker() -> None:
    global _WORKER_CONN, _WORKER_PROCESS
    if model_worker_running():
        return

    # chuthich: Dung spawn context de tach RAM model khoi web process tren server Windows/Linux.
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=worker_loop, args=(child_conn,), daemon=True)
    process.start()
    child_conn.close()
    _WORKER_CONN = parent_conn
    _WORKER_PROCESS = process


def stop_model_worker() -> None:
    global _WORKER_CONN, _WORKER_PROCESS
    process = _WORKER_PROCESS
    conn = _WORKER_CONN
    _WORKER_CONN = None
    _WORKER_PROCESS = None

    if conn is not None:
        try:
            if process is not None and process.is_alive():
                conn.send({"command": "stop"})
        except Exception:
            pass
        finally:
            conn.close()

    if process is not None and process.is_alive():
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)


def clear_model_cache_and_worker() -> None:
    # chuthich: Parent process khong giu model; ham nay dam bao worker bi tat han.
    stop_model_worker()
    clear_model_cache()


def analyze_upload(input_ref: InputRef, data: bytes) -> dict[str, Any]:
    # chuthich: Web process chi gui anh sang worker, khong load model vao RAM cua web.
    with _WORKER_LOCK:
        start_model_worker()
        if _WORKER_CONN is None:
            raise RuntimeError("Không khởi động được model worker.")

        try:
            _WORKER_CONN.send(
                {
                    "command": "analyze",
                    "input_ref": {"name": input_ref.name, "kind": input_ref.kind},
                    "data": data,
                }
            )
            timeout = safe_env_float(WORKER_TIMEOUT_ENV, 120.0)
            if not _WORKER_CONN.poll(timeout):
                stop_model_worker()
                raise RuntimeError("Model worker phản hồi quá lâu, đã được khởi động lại.")

            response = _WORKER_CONN.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            stop_model_worker()
            raise RuntimeError("Model worker đã dừng bất thường.") from exc

        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Model worker lỗi."))
        return response["result"]
