from __future__ import annotations

import webbrowser
from threading import Timer

from app.web import create_app


def main() -> None:
    url = "http://127.0.0.1:5001"
    app = create_app()
    Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"AI Smart Drainage is starting at {url}")
    print("Use Start Camera to grant browser webcam access.")
    app.run(host="127.0.0.1", port=5001, threaded=True, debug=False)


if __name__ == "__main__":
    main()
