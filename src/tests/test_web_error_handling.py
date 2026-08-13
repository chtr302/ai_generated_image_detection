from __future__ import annotations

import os
import unittest
from io import BytesIO
from unittest.mock import patch

from src.web.app import MODEL_STATE, create_app


def clear_key_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("AIGID_MODEL_CONTROL_KEY", None)
    return env


class WebErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, clear_key_env(), clear=True)
        self.env_patch.start()
        MODEL_STATE["enabled"] = True
        self.client = create_app().test_client()

    def tearDown(self) -> None:
        MODEL_STATE["enabled"] = True
        self.env_patch.stop()

    def test_model_off_blocks_load_url_and_analyze(self) -> None:
        MODEL_STATE["enabled"] = False

        load_response = self.client.post("/api/load-url", data={"url": "https://example.com/image.png"})
        self.assertEqual(load_response.status_code, 503)
        self.assertEqual(load_response.json["mode"], "off")

        analyze_response = self.client.post("/api/analyze", data={})
        self.assertEqual(analyze_response.status_code, 503)
        self.assertEqual(analyze_response.json["mode"], "off")

    def test_load_url_rejects_invalid_url(self) -> None:
        response = self.client.post("/api/load-url", data={"url": "not-a-url"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_analyze_rejects_empty_input(self) -> None:
        response = self.client.post("/api/analyze", data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_analyze_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={"image": (BytesIO(b"not an image"), "document.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)


if __name__ == "__main__":
    unittest.main()
