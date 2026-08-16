from src.web.app import app


def main():
    import os

    host = os.getenv("AIGID_HOST", "0.0.0.0")
    port = int(os.getenv("AIGID_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
