import threading

from app.classification import Prediction
from app.worker import PlasticWorker


def update_when_complete(results, completed):
    def update(key, prediction, reason):
        results.append((key, prediction, reason))
        completed.set()

    return update


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

    worker = PlasticWorker(
        update,
        custom_classifier=FakeCustomClassifier(
            Prediction(
                "uncertain",
                "Custom model is unavailable.",
                None,
                0.0,
                "custom_model",
                "model_unavailable",
            )
        ),
        min_free_ram_gib=5,
    )
    assert worker._classifier is None
    assert worker.submit("plastic-1", b"plastic-crop")
    assert completed.wait(3)
    assert worker._classifier is not None
    assert results[0][0] == "plastic-1"
    assert results[0][1].label == "recyclable"
    assert results[0][1].source == "qwen"
    assert results[0][1].reason == (
        "Custom model is unavailable.; Qwen fallback: Clean rigid plastic."
    )
    assert results[0][2] is None


class FakeCustomClassifier:
    def __init__(self, prediction, events=None):
        self.prediction = prediction
        self.events = events if events is not None else []

    def classify(self, crop):
        self.events.append(("custom", crop))
        return self.prediction

    def release(self):
        self.events.append(("release", None))


def test_worker_uses_custom_result_without_starting_qwen(monkeypatch):
    completed = threading.Event()
    results = []
    custom = FakeCustomClassifier(
        Prediction("non_recyclable", "Flexible film.", 0.93, 0.2, "custom_model")
    )

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(6 * 1024 ** 3)})(),
    )
    monkeypatch.setattr(
        "app.worker.OllamaPlasticClassifier",
        lambda: (_ for _ in ()).throw(AssertionError("Qwen should not be initialized.")),
    )
    worker = PlasticWorker(
        update_when_complete(results, completed),
        custom_classifier=custom,
        min_free_ram_gib=5,
    )

    assert worker.submit("plastic-1", b"crop")
    assert completed.wait(3)
    assert results[0][1].source == "custom_model"
    assert results[0][1].label == "non_recyclable"
    assert not any(event[0] == "release" for event in custom.events)


def test_worker_releases_uncertain_custom_model_before_qwen_fallback(monkeypatch):
    completed = threading.Event()
    results = []
    events = []
    custom = FakeCustomClassifier(
        Prediction(
            "uncertain",
            "Top confidence below threshold.",
            0.51,
            0.2,
            "custom_model",
            "low_confidence",
        ),
        events,
    )

    class FakeQwen:
        def classify(self, crop):
            assert events[-1][0] == "release"
            assert crop == b"crop"
            return Prediction("uncertain", "Qwen is unsure.", 0.4, 0.3)

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(6 * 1024 ** 3)})(),
    )
    worker = PlasticWorker(
        update_when_complete(results, completed),
        classifier=FakeQwen(),
        custom_classifier=custom,
        min_free_ram_gib=5,
    )

    assert worker.submit("plastic-1", b"crop")
    assert completed.wait(3)
    prediction = results[0][1]
    assert prediction.label == "uncertain"
    assert prediction.source == "qwen"
    assert prediction.inference_seconds == 0.5
    assert "Top confidence below threshold." in prediction.reason
    assert "Qwen is unsure." in prediction.reason


def test_worker_skips_qwen_for_invalid_custom_crop(monkeypatch):
    completed = threading.Event()
    results = []
    custom = FakeCustomClassifier(
        Prediction("uncertain", "Invalid crop.", None, 0.0, "none", "invalid_image")
    )

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(6 * 1024 ** 3)})(),
    )
    monkeypatch.setattr(
        "app.worker.OllamaPlasticClassifier",
        lambda: (_ for _ in ()).throw(AssertionError("Qwen should not run on an invalid crop.")),
    )
    worker = PlasticWorker(
        update_when_complete(results, completed),
        custom_classifier=custom,
        min_free_ram_gib=5,
    )

    assert worker.submit("plastic-1", b"bad-image")
    assert completed.wait(3)
    assert results[0][1].error_code == "invalid_image"
    assert results[0][2] is None


def test_worker_does_not_run_custom_or_qwen_when_ram_is_low():
    completed = threading.Event()
    results = []

    class NeverCustom:
        def classify(self, crop):
            raise AssertionError("Custom classifier must not run when RAM is low.")

        def release(self):
            raise AssertionError("No model should be loaded.")

    worker = PlasticWorker(
        update_when_complete(results, completed),
        custom_classifier=NeverCustom(),
        min_free_ram_gib=float("inf"),
    )

    assert worker.submit("plastic-1", b"crop")
    assert completed.wait(3)
    assert results[0][1] is None
    assert "Insufficient free RAM" in results[0][2]


def test_worker_rechecks_ram_before_qwen_after_custom_inference(monkeypatch):
    completed = threading.Event()
    results = []
    memory_values = iter((6, 4))

    class FakeQwen:
        def classify(self, crop):
            raise AssertionError("Qwen must not run when RAM falls below the limit.")

    monkeypatch.setattr(
        "app.worker.psutil.virtual_memory",
        lambda: type("Memory", (), {"available": int(next(memory_values) * 1024 ** 3)})(),
    )
    worker = PlasticWorker(
        update_when_complete(results, completed),
        classifier=FakeQwen(),
        custom_classifier=FakeCustomClassifier(
            Prediction(
                "uncertain",
                "Custom confidence is low.",
                0.52,
                0.2,
                "custom_model",
                "low_confidence",
            )
        ),
        min_free_ram_gib=5,
    )

    assert worker.submit("plastic-1", b"crop")
    assert completed.wait(3)
    assert results[0][1].label == "uncertain"
    assert results[0][1].confidence == 0.52
    assert "Insufficient free RAM" in results[0][1].reason
