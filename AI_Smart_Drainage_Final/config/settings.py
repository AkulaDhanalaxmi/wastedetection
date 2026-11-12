from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = Path(os.getenv("SMART_DRAINAGE_MODEL_PATH", str(ROOT / "models" / "best.pt"))
                  ).expanduser().resolve()

DETECTION_CONFIDENCE = float(os.getenv("SMART_DRAINAGE_DETECTION_CONFIDENCE", "0.35"))
YOLO_IMAGE_SIZE = int(os.getenv("SMART_DRAINAGE_YOLO_IMAGE_SIZE", "640"))
TRACKER_CONFIG = os.getenv("SMART_DRAINAGE_TRACKER", "bytetrack.yaml")

OLLAMA_URL = os.getenv("SMART_DRAINAGE_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("SMART_DRAINAGE_OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("SMART_DRAINAGE_OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_MAX_IMAGE_SIDE = int(os.getenv("SMART_DRAINAGE_OLLAMA_MAX_IMAGE_SIDE", "512"))
OLLAMA_KEEP_ALIVE = os.getenv("SMART_DRAINAGE_OLLAMA_KEEP_ALIVE", "0")
MIN_FREE_RAM_GIB = float(os.getenv("SMART_DRAINAGE_MIN_FREE_RAM_GIB", "5"))
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_INPUT_IMAGE_SIDE = 1280
MAX_PENDING_CLASSIFICATIONS = 4
