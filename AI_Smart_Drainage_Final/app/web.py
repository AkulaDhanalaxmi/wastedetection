from __future__ import annotations

import base64
import threading
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from app.counts import SessionCounts
from app.detector import RawDetection, WasteDetector
from config.settings import MAX_INPUT_IMAGE_SIDE, MAX_UPLOAD_BYTES
from app.worker import PlasticWorker


def create_app(
    *,
    detector: WasteDetector | None = None,
    worker: PlasticWorker | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    detector = detector or WasteDetector()
    counts = SessionCounts()
    snapshot_lock = threading.Lock()
    latest_detections: list[dict] = []
    camera_active = False
    camera_session = uuid.uuid4().hex
    worker = worker or PlasticWorker(counts.apply_prediction)
    app.extensions["waste_detector"] = detector
    app.extensions["waste_counts"] = counts
    app.extensions["plastic_worker"] = worker

    def process_detections(
        detections: list[RawDetection],
        *,
        id_prefix: str,
        tracking: bool,
    ) -> list[dict]:
        output = []
        for index, detection in enumerate(detections):
            key = (
                f"{id_prefix}:{counts.epoch}:{detection.track_id}"
                if tracking and detection.track_id is not None
                else f"{id_prefix}:{counts.epoch}:{index}"
            )
            _, queue_classification = counts.observe(
                key if (tracking and detection.track_id is not None) or not tracking else None,
                detection.material,
                detection.confidence,
                detection.crop_jpeg,
                confirm_immediately=not tracking,
            )
            if queue_classification and detection.crop_jpeg is not None:
                worker.submit(key, detection.crop_jpeg)
            item = counts.item(key)
            if item is None:
                item = {
                    "track_id": None,
                    "object_name": detection.material,
                    "material": detection.material,
                    "category": (
                        "PLASTIC_MANUAL_REVIEW"
                        if detection.material == "Plastic"
                        else detection.material.upper()
                    ),
                    "detection_confidence": detection.confidence,
                    "classification_confidence": None,
                    "classification_inference_seconds": None,
                    "reason": "YOLO tracking ID unavailable; not added to unique counts.",
                    "classification_state": "not_counted",
                }
            item["box"] = list(detection.box)
            output.append(item)
        return output

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        with snapshot_lock:
            current_detections = []
            for detection in latest_detections:
                updated = counts.item(detection["track_id"]) if detection["track_id"] else None
                current_detections.append(
                    {**detection, **updated} if updated is not None else detection
                )
            active = camera_active
        return jsonify({"camera_active": active, **counts.snapshot(), "detections": current_detections})

    @app.post("/api/camera/start")
    def camera_start():
        nonlocal camera_active, camera_session
        try:
            detector.load()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 500
        with snapshot_lock:
            if not camera_active:
                camera_session = uuid.uuid4().hex
            camera_active = True
        return jsonify({"camera_active": True, "message": "Camera analysis is ready."})

    @app.post("/api/camera/stop")
    def camera_stop():
        nonlocal camera_active, latest_detections
        with snapshot_lock:
            camera_active = False
        detector.reset_tracker()
        with snapshot_lock:
            latest_detections = []
        return jsonify({"camera_active": False, **counts.snapshot()})

    @app.post("/api/counts/reset")
    def reset_counts():
        counts.reset()
        detector.reset_tracker()
        with snapshot_lock:
            latest_detections.clear()
        return jsonify(counts.snapshot())

    @app.post("/api/frame")
    def camera_frame():
        nonlocal latest_detections
        with snapshot_lock:
            active = camera_active
            current_session = camera_session
        if not active:
            return jsonify({"error": "Start the camera before sending frames."}), 409
        raw = request.get_data(cache=False)
        frame = _decode_image(raw)
        if frame is None:
            return jsonify({"error": "Could not decode camera frame."}), 400
        try:
            found = detector.detect(frame, tracking=True)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 500
        result = process_detections(
            found,
            id_prefix=f"camera:{current_session}",
            tracking=True,
        )
        with snapshot_lock:
            latest_detections = result
        return jsonify({"detections": result, **counts.snapshot()})

    @app.post("/api/upload")
    def upload_image():
        nonlocal latest_detections
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Select an image to analyze."}), 400
        raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "Image exceeds the 12 MiB upload limit."}), 413
        frame = _decode_image(raw)
        del raw
        if frame is None:
            return jsonify({"error": "Uploaded file is not a supported image."}), 400
        try:
            found = detector.detect(frame, tracking=False)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 500
        upload_id = f"upload:{uuid.uuid4().hex}"
        detections = process_detections(found, id_prefix=upload_id, tracking=False)
        with snapshot_lock:
            latest_detections = detections
        annotated = _annotate(frame, detections)
        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            return jsonify({"error": "Could not render annotated image."}), 500
        return jsonify({
            "annotated_image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii"),
            "detections": detections,
            **counts.snapshot(),
        })

    return app


def _decode_image(raw: bytes) -> np.ndarray | None:
    if not raw:
        return None
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if max(height, width) <= MAX_INPUT_IMAGE_SIDE:
        return frame
    scale = MAX_INPUT_IMAGE_SIDE / max(height, width)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    del frame
    return resized


def _annotate(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    colors = {
        "PLASTIC_RECYCLABLE": (22, 163, 74),
        "PLASTIC_NON_RECYCLABLE": (220, 38, 38),
        "PLASTIC_MANUAL_REVIEW": (217, 119, 6),
        "METAL": (37, 99, 235),
        "ORGANIC": (146, 52, 33),
    }
    image = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = colors.get(item["category"], (128, 128, 128))
        label = f'{item["material"]} | {item["category"]}'
        label += f' | det {item["detection_confidence"]:.2f}'
        if item["classification_confidence"] is not None:
            label += f' | Qwen {item["classification_confidence"]:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return image
