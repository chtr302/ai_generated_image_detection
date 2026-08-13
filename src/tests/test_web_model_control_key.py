from __future__ import annotations

import os
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from src.web.app import create_app
from src.web.services import inference as inference_service


def clear_key_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("AIGID_MODEL_CONTROL_KEY", None)
    return env


def tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class ModelControlKeyTests(unittest.TestCase):
    def tearDown(self) -> None:
        inference_service.clear_model_cache_and_worker()

    def test_model_control_is_open_when_key_is_not_configured(self) -> None:
        with patch.dict(os.environ, clear_key_env(), clear=True):
            client = create_app().test_client()

            test_key_response = client.post("/api/model-control-key/test")
            self.assertEqual(test_key_response.status_code, 200)
            self.assertIs(test_key_response.json["valid"], True)
            self.assertIs(test_key_response.json["configured"], False)

            off_response = client.post("/api/model-state", json={"enabled": False})
            self.assertEqual(off_response.status_code, 200)
            self.assertIs(off_response.json["enabled"], False)

            on_response = client.post("/api/model-state", json={"enabled": True})
            self.assertEqual(on_response.status_code, 200)
            self.assertIs(on_response.json["enabled"], True)

    def test_model_control_key_missing_wrong_and_valid_paths(self) -> None:
        env = clear_key_env()
        env["AIGID_MODEL_CONTROL_KEY"] = "secret-key"
        with patch.dict(os.environ, env, clear=True):
            client = create_app().test_client()

            missing_response = client.post("/api/model-state", json={"enabled": False})
            self.assertEqual(missing_response.status_code, 401)
            self.assertEqual(missing_response.json["error_code"], "missing_model_control_key")

            wrong_response = client.post(
                "/api/model-state",
                json={"enabled": False},
                headers={"X-Model-Control-Key": "wrong-key"},
            )
            self.assertEqual(wrong_response.status_code, 403)
            self.assertEqual(wrong_response.json["error_code"], "invalid_model_control_key")

            test_key_response = client.post(
                "/api/model-control-key/test",
                headers={"X-Model-Control-Key": "secret-key"},
            )
            self.assertEqual(test_key_response.status_code, 200)
            self.assertIs(test_key_response.json["valid"], True)
            self.assertIs(test_key_response.json["configured"], True)

            off_response = client.post(
                "/api/model-state",
                json={"enabled": False},
                headers={"X-Model-Control-Key": "secret-key"},
            )
            self.assertEqual(off_response.status_code, 200)
            self.assertIs(off_response.json["enabled"], False)

            analyze_response = client.post("/api/analyze", data={})
            self.assertEqual(analyze_response.status_code, 503)

            on_response = client.post(
                "/api/model-state",
                json={"enabled": True},
                headers={"X-Model-Control-Key": "secret-key"},
            )
            self.assertEqual(on_response.status_code, 200)
            self.assertIs(on_response.json["enabled"], True)

    def test_model_control_rejects_invalid_payload(self) -> None:
        with patch.dict(os.environ, clear_key_env(), clear=True):
            client = create_app().test_client()

            missing_enabled_response = client.post("/api/model-state", json={})
            self.assertEqual(missing_enabled_response.status_code, 400)
            self.assertEqual(missing_enabled_response.json["error_code"], "missing_enabled")

            invalid_enabled_response = client.post("/api/model-state", json={"enabled": "false"})
            self.assertEqual(invalid_enabled_response.status_code, 400)
            self.assertEqual(invalid_enabled_response.json["error_code"], "invalid_enabled")

    def test_analyze_starts_worker_off_stops_worker_and_on_is_lazy(self) -> None:
        env = clear_key_env()
        env["AIGID_WORKER_TIMEOUT_SECONDS"] = "10"
        with patch.dict(os.environ, env, clear=True):
            client = create_app().test_client()

            before_detail = client.get("/api/model-detail")
            self.assertEqual(before_detail.status_code, 200)
            self.assertIs(before_detail.json["worker_running"], False)
            self.assertIsNone(before_detail.json["worker_pid"])

            analyze_response = client.post(
                "/api/analyze",
                data={"image": (BytesIO(tiny_png()), "tiny.png")},
                content_type="multipart/form-data",
            )
            self.assertEqual(analyze_response.status_code, 200)
            self.assertIn(analyze_response.json["final_label"], {"Real", "AI-generated"})

            loaded_detail = client.get("/api/model-detail")
            self.assertEqual(loaded_detail.status_code, 200)
            self.assertIs(loaded_detail.json["worker_running"], True)
            self.assertIsInstance(loaded_detail.json["worker_pid"], int)

            off_response = client.post("/api/model-state", json={"enabled": False})
            self.assertEqual(off_response.status_code, 200)
            self.assertIs(off_response.json["enabled"], False)

            off_detail = client.get("/api/model-detail")
            self.assertEqual(off_detail.status_code, 200)
            self.assertIs(off_detail.json["worker_running"], False)
            self.assertIsNone(off_detail.json["worker_pid"])

            on_response = client.post("/api/model-state", json={"enabled": True})
            self.assertEqual(on_response.status_code, 200)
            self.assertIs(on_response.json["enabled"], True)

            on_detail = client.get("/api/model-detail")
            self.assertEqual(on_detail.status_code, 200)
            self.assertIs(on_detail.json["worker_running"], False)
            self.assertIsNone(on_detail.json["worker_pid"])


if __name__ == "__main__":
    unittest.main()
