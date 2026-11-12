# AI Smart Drainage Final

A compact local Flask app that uses the included YOLO11 checkpoint to detect
Plastic, Organic, and Metal from a browser webcam. The dashboard sends webcam
frames to the local Flask backend continuously while the camera is running.
Plastic crops are first checked by the optional trained custom classifier from
`plastic_classification_module`. If its weights are not present or it abstains
at its configured confidence threshold, the existing local Ollama/Qwen2.5-VL
classifier is used as a fallback. Uncertain results remain manual review;
unsupported labels are rejected.

## Run on Windows

Requirements: Python 3.10+, Ollama, and the local `qwen2.5vl:3b` model.
The YOLO checkpoint is included at `models/best.pt`; Ollama manages its model
outside this folder.

```powershell
ollama list
```
Confirm `qwen2.5vl:3b` is listed, then double-click `run.bat`. The first run
creates `.venv` and installs `requirements.txt`; the browser opens at
`http://127.0.0.1:5001`. Click **Start Camera**, allow webcam access, and hold
an item in view. The camera view reports when no supported item is detected,
shows YOLO detection confidence as a percentage, and updates Plastic results
when the local classifier finishes. Image upload remains an optional test path;
it is not needed for live detection.

## What the models can and cannot tell you

- The included YOLO checkpoint contains exactly three classes: `Plastic`,
  `Metal`, and `Organic`. It identifies the material category, not specific
  product names such as bottle, can, or food wrapper.
- Plastic recyclability is a separate visual judgment. When a trained custom
  model is available, its calibrated confidence threshold is honored; low
  confidence, missing weights, or model load failures use local Qwen2.5-VL.
  Ambiguous crops, failures, and unsupported responses stay in manual review.
- The custom model is optional. Its default checkpoint is
  `../AI_SMART_DRAINAGE_PLASTIC_CLASSIFICATION/plastic_classification_module/runs/plastic/default/best.pt`;
  until that checkpoint exists, the current Qwen workflow continues unchanged.
  Override the module or checkpoint location with the optional
  `SMART_DRAINAGE_PLASTIC_MODULE_PATH` and
  `SMART_DRAINAGE_PLASTIC_MODEL_PATH` settings.
- To run a trained PyTorch checkpoint, install the plastic module's
  `requirements.txt` into the app's Python environment. This keeps the large
  PyTorch dependencies optional for users who have not trained the classifier.
- Qwen's displayed confidence is model-reported and is not calibrated to
  real-world accuracy. Appearance alone cannot confirm resin type, contamination
  hidden from view, or local recycling rules. Treat the result as guidance, not
  a guarantee.
- The displayed detection percentage is YOLO's confidence for the detected
  material class. The configured minimum detection threshold is 0.35; objects
  below that threshold are not reported as detections.
- Camera access is provided by the browser and normally requires localhost or a
  secure HTTPS context. If access is denied or no camera is available, check
  browser permissions and close other applications using the webcam.

## Memory behavior

- One CPU YOLO model is loaded lazily and reused.
- The custom plastic model loads only when its trained checkpoint exists.
- Custom-model weights are released before a Qwen fallback is requested.
- The single Qwen worker starts only when a Plastic crop needs fallback classification.
- Plastic crops and incoming images are resized before downstream processing.
- At most four small crops wait in the classification queue; excess items are
  reported for manual review rather than retaining unbounded image data.
- Qwen is asked to unload after each request (`keep_alive=0`).
- Classification is refused unless at least 5 GiB RAM is available. The limit
  is deliberate and must not be reduced to force inference on low memory.

Qwen is about 3.2 GB on disk and requires additional runtime memory. With only
8 GB installed RAM, other Windows applications, Python, OpenCV, and YOLO may
leave too little headroom for Qwen. If the 5 GiB guard refuses classification,
the app will show manual review instead of attempting an unsafe request.

## Configuration

Copy `.env.example` to `.env` to override settings. The classifier defaults
are Qwen2.5-VL 3B, 512-pixel maximum crop side, unload-after-request, and a
5 GiB minimum available-RAM check. Detection uses the included `models/best.pt`.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```

These are offline unit and integration tests using controlled fake detections
and mocked classifier responses. They verify application logic, not the
accuracy of a real YOLO or Qwen prediction. Real model inference requires
sufficient free RAM and should be tested on the target laptop.
