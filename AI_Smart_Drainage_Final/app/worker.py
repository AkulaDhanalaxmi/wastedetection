from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import psutil

from app.classification import ClassificationError, OllamaPlasticClassifier, Prediction
from config.settings import MAX_PENDING_CLASSIFICATIONS, MIN_FREE_RAM_GIB


@dataclass(frozen=True)
class ClassificationJob:
    item_key: str
    crop: bytes


class PlasticWorker:
    def __init__(
        self,
        update_result: Callable[[str, Prediction | None, str | None], None],
        *,
        classifier: OllamaPlasticClassifier | None = None,
        min_free_ram_gib: float = MIN_FREE_RAM_GIB,
        autostart: bool = True,
    ) -> None:
        self._update_result = update_result
        self._classifier = classifier
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
                if self._classifier is None:
                    self._classifier = OllamaPlasticClassifier()
                prediction = self._classifier.classify(job.crop)
            except ClassificationError as exc:
                self._update_result(job.item_key, None, f"{exc}; manual review required.")
            except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
                self._update_result(
                    job.item_key,
                    None,
                    f"Classifier failed ({type(exc).__name__}); manual review required.",
                )
            else:
                self._update_result(job.item_key, prediction, None)
            finally:
                self._jobs.task_done()
                del job

    @property
    def pending(self) -> int:
        return self._jobs.qsize()
