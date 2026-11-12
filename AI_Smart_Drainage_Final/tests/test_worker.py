import threading

from app.classification import Prediction
from app.worker import PlasticWorker


class FakeClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, crop):
        self.calls += 1
        raise AssertionError("Classifier must not run when RAM is below the safety limit.")


def test_worker_returns_manual_review_without_loading_qwen_when_ram_is_low(monkeypatch):
    classifier = FakeClassifier()
    completed = threading.Event()
    results = []

    def update(key, prediction, reason):
        results.append((key, prediction, reason))
        completed.set()

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(2 * 1024 ** 3)})(),
    )
    worker = PlasticWorker(
        update,
        classifier=classifier,
        min_free_ram_gib=5,
    )
    assert worker.submit("plastic-1", b"crop")
    assert completed.wait(3)
    assert classifier.calls == 0
    assert results[0][0] == "plastic-1"
    assert results[0][1] is None
    assert "Insufficient free RAM" in results[0][2]


def test_worker_creates_classifier_only_after_plastic_job_and_sufficient_ram(monkeypatch):
    completed = threading.Event()
    results = []

    class FakeClassifier:
        def classify(self, crop):
            assert crop == b"plastic-crop"
            return Prediction("recyclable", "Clean rigid plastic.", 0.9, 0.1)

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(6 * 1024 ** 3)})(),
    )
    monkeypatch.setattr(
        "app.worker.OllamaPlasticClassifier",
        lambda: FakeClassifier(),
    )

    def update(key, prediction, reason):
        results.append((key, prediction, reason))
        completed.set()

    worker = PlasticWorker(update, min_free_ram_gib=5)
    assert worker._classifier is None
    assert worker.submit("plastic-1", b"plastic-crop")
    assert completed.wait(3)
    assert worker._classifier is not None
    assert results == [
        ("plastic-1", Prediction("recyclable", "Clean rigid plastic.", 0.9, 0.1), None)
    ]
