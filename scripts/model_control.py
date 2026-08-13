from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:5000"
KEY_ENV = "AIGID_MODEL_CONTROL_KEY"
KEY_HEADER = "X-Model-Control-Key"


def default_key() -> str:
    return os.getenv(KEY_ENV, "").strip()


def request_json(url: str, method: str = "GET", payload: dict[str, object] | None = None, key: str = "") -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if key:
        headers[KEY_HEADER] = key

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = {"error": body or exc.reason}
        raise SystemExit(f"HTTP {exc.code}: {error_payload.get('error', error_payload)}") from exc
    except URLError as exc:
        raise SystemExit(f"Khong ket noi duoc server: {exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Bat/tat model web demo bang command line.")
    parser.set_defaults(key=default_key())
    parser.add_argument("action", choices=["status", "on", "off", "test-key"], help="Trang thai can xem, dat, hoac test key.")
    parser.add_argument("--base-url", default=os.getenv("AIGID_WEB_URL", DEFAULT_BASE_URL), help="URL Flask app.")
    parser.add_argument(
        "--key",
        help=f"Model control key. Mac dinh doc tu {KEY_ENV}.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.action == "status":
        result = request_json(f"{base_url}/api/model-state")
    elif args.action == "test-key":
        result = request_json(f"{base_url}/api/model-control-key/test", method="POST", key=args.key)
        print(f"key_valid={result.get('valid')} configured={result.get('configured')} required={result.get('required')}")
        print(result.get("message", ""))
        return
    else:
        result = request_json(
            f"{base_url}/api/model-state",
            method="POST",
            payload={"enabled": args.action == "on"},
            key=args.key,
        )

    print(f"model={result.get('mode')} enabled={result.get('enabled')}")
    print(result.get("message", ""))


if __name__ == "__main__":
    main()
