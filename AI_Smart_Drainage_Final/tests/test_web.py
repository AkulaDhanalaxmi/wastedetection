import io

import cv2
import numpy as np

from app.detector import RawDetection
from app.classification import Prediction
from app.web import create_app


class FakeDetector:
    def __init__(self):
        self.loads = 0
        self.resets = 0

    def load(self):
        self.loads += 1

    def reset_tracker(self):
        self.resets += 1

    def detect(self, frame, *, tracking):
        if tracking:
            return [
                RawDetection("17", "Plastic", 0.91, (2, 2, 30, 30), b"plastic-crop"),
                RawDetection("18", "Metal", 0.88, (32, 2, 60, 30), None),
                RawDetection("19", "Organic", 0.82, (62, 2, 90, 30), None),
            ]
        return [
            RawDetection(None, "Plastic", 0.91, (2, 2, 30, 30), b"plastic-crop"),
            RawDetection(None, "Metal", 0.88, (32, 2, 60, 30), None),
            RawDetection(None, "Organic", 0.82, (62, 2, 90, 30), None),
        ]


class FakeWorker:
    def __init__(self):
        self.jobs = []

    def submit(self, key, crop):
        self.jobs.append((key, crop))
        return True


def jpeg_bytes():
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    success, data = cv2.imencode(".jpg", image)
    assert success
    return data.tobytes()


def test_browser_dashboard_and_image_upload_flow():
    detector = FakeDetector()
    worker = FakeWorker()
    app = create_app(detector=detector, worker=worker)
    app.config["TESTING"] = True
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Start Camera" in page.data
    assert b"Plastic manual review" in page.data
    assert b"camera-message" in page.data
    assert b"Detection confidence" in page.data

    response = client.post(
        "/api/upload",
        data={"image": (io.BytesIO(jpeg_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    result = response.get_json()
    assert len(result["detections"]) == 3
    assert result["total"] == 3
    assert result["plastic_total"] == result["plastic_manual_review"] == 1
    assert result["metal_total"] == 1
    assert result["organic_total"] == 1
    assert result["detections"][0]["detection_confidence"] == 0.91
    assert result["detections"][0]["classification_confidence"] is None
    assert result["annotated_image"].startswith("data:image/jpeg;base64,")
    assert len(worker.jobs) == 1

    counts = app.extensions["waste_counts"]
    item_key = worker.jobs[0][0]
    from app.classification import Prediction

    counts.apply_prediction(
        item_key,
        Prediction("recyclable", "Clean rigid plastic.", 0.77, 2.8),
    )
    final = client.get("/api/state").get_json()
    assert final["plastic_total"] == final["plastic_recyclable"] == 1
    assert final["plastic_manual_review"] == 0
    assert final["total"] == 3
    plastic_row = next(row for row in final["detections"] if row["material"] == "Plastic")
    assert plastic_row["category"] == "PLASTIC_RECYCLABLE"
    assert plastic_row["classification_inference_seconds"] == 2.8


def test_webcam_frame_tracking_counts_only_once_and_stop_resets_tracker():
    detector = FakeDetector()
    worker = FakeWorker()
    app = create_app(detector=detector, worker=worker)
    app.config["TESTING"] = True
    client = app.test_client()

    assert client.post("/api/camera/start").status_code == 200
    frame = jpeg_bytes()
    for _ in range(6):
        response = client.post(
            "/api/frame",
            data=frame,
            content_type="application/octet-stream",
        )
        assert response.status_code == 200
    state = client.get("/api/state").get_json()
    assert state["camera_active"]
    assert state["total"] == 3
    assert state["plastic_total"] == 1
    assert state["metal_total"] == 1
    assert state["organic_total"] == 1
    assert len(worker.jobs) == 1

    counts = app.extensions["waste_counts"]
    counts.apply_prediction(
        worker.jobs[0][0],
        Prediction("non_recyclable", "Flexible film is visible.", 0.74, 1.2),
    )
    updated = client.get("/api/state").get_json()
    plastic_row = next(row for row in updated["detections"] if row["material"] == "Plastic")
    assert plastic_row["category"] == "PLASTIC_NON_RECYCLABLE"
    assert plastic_row["classification_state"] == "classified"

    stopped = client.post("/api/camera/stop")
    assert stopped.status_code == 200
    assert not stopped.get_json()["camera_active"]
    assert detector.resets == 1
    assert client.post(
        "/api/frame",
        data=frame,
        content_type="application/octet-stream",
    ).status_code == 409
