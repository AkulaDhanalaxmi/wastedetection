from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import psutil

from app.classification import ClassificationError, OllamaPlasticClassifier, Prediction
from app.custom_classifier import CustomPlasticClassifier
from config.settings import MAX_PENDING_CLASSIFICATIONS, MIN_FREE_RAM_GIB


class _Classifier(Protocol):
    def classify(self, crop: bytes) -> Prediction: ...


class _CustomClassifier(_Classifier, Protocol):
    def release(self) -> None: ...


@dataclass(frozen=True)
class ClassificationJob:
    item_key: str
    crop: bytes


class PlasticWorker:
    def __init__(
        self,
        update_result: Callable[[str, Prediction | None, str | None], None],
        *,
        classifier: _Classifier | None = None,
        custom_classifier: _CustomClassifier | None = None,
        min_free_ram_gib: float = MIN_FREE_RAM_GIB,
        autostart: bool = True,
    ) -> None:
        self._update_result = update_result
        self._classifier = classifier
        self._custom_classifier = custom_classifier or CustomPlasticClassifier()
        self._min_free_ram_gib = min_free_ram_gib
        self._jobs: queue.Queue[ClassificationJob] = queue.Queue(
            maxsize=MAX_PENDING_CLASSIFICATIONS
        )
        self._thread = threading.Thread(
            target=self._run,
            name="plastic-classifier",
            daemon=True,
        )
        self._thread_lock = threading.Lock()
        self._autostart = autostart

    def submit(self, item_key: str, crop: bytes) -> bool:
        try:
            self._jobs.put_nowait(ClassificationJob(item_key, crop))
        except queue.Full:
            self._update_result(item_key, None, "Classifier queue is full; manual review required.")
            return False
        if self._autostart and not self._thread.is_alive():
            with self._thread_lock:
                if not self._thread.is_alive():
                    self._thread.start()
        return True

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            custom_prediction: Prediction | None = None
            try:
                free_gib = psutil.virtual_memory().available / (1024 ** 3)
                if free_gib < self._min_free_ram_gib:
                    self._update_result(
                        job.item_key,
                        None,
                        f"Insufficient free RAM ({free_gib:.2f} GiB); "
                        f"{self._min_free_ram_gib:.1f} GiB required. Manual review.",
                    )
                    continue
                custom_prediction = self._custom_classifier.classify(job.crop)
                if custom_prediction.error_code in {
                    "empty_crop",
                    "invalid_image",
                    "crop_too_small",
                    "blurry_crop",
                }:
                    self._update_result(job.item_key, custom_prediction, None)
                    continue
                if (
                    custom_prediction.error_code not in {"model_unavailable", "low_confidence"}
                    and custom_prediction.label != "uncertain"
                ):
                    self._update_result(job.item_key, custom_prediction, None)
                    continue

                self._custom_classifier.release()
                free_gib = psutil.virtual_memory().available / (1024 ** 3)
                if free_gib < self._min_free_ram_gib:
                    self._update_result(
                        job.item_key,
                        replace(
                            custom_prediction,
                            reason=(
                                f"{custom_prediction.reason}; Insufficient free RAM "
                                f"({free_gib:.2f} GiB) for Qwen fallback; "
                                "manual review required."
                            ),
                        ),
                        None,
                    )
                    continue
                if self._classifier is None:
                    self._classifier = OllamaPlasticClassifier()
                prediction = self._classifier.classify(job.crop)
                prediction = replace(
                    prediction,
                    reason=(
                        f"{custom_prediction.reason}; Qwen fallback: {prediction.reason}"
                    ),
                    inference_seconds=(
                        custom_prediction.inference_seconds + prediction.inference_seconds
                    ),
                    source="qwen",
                )
            except ClassificationError as exc:
                if custom_prediction is not None and custom_prediction.error_code == "low_confidence":
                    self._update_result(
                        job.item_key,
                        replace(
                            custom_prediction,
                            reason=f"{custom_prediction.reason}; Qwen fallback failed: {exc}",
                        ),
                        None,
                    )
                else:
                    reason = f"{exc}; manual review required."
                    if custom_prediction is not None:
                        reason = f"{custom_prediction.reason}; {reason}"
                    self._update_result(job.item_key, None, reason)
            except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
                if custom_prediction is not None and custom_prediction.error_code == "low_confidence":
                    self._update_result(
                        job.item_key,
                        replace(
                            custom_prediction,
                            reason=(
                                f"{custom_prediction.reason}; Qwen fallback failed "
                                f"({type(exc).__name__}): {exc}"
                            ),
                        ),
                        None,
                    )
                    continue
                reason = (
                    f"Classifier failed ({type(exc).__name__}); manual review required."
                )
                if custom_prediction is not None:
                    reason = f"{custom_prediction.reason}; {reason}"
                self._update_result(
                    job.item_key,
                    None,
                    reason,
                )
            else:
                self._update_result(job.item_key, prediction, None)
            finally:
                self._jobs.task_done()
                del job

    @property
    def pending(self) -> int:
        return self._jobs.qsize()
