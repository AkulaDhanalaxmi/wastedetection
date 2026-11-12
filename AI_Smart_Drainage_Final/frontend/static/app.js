const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const resetButton = document.getElementById("reset-counts");
const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");
const statusText = document.getElementById("status");
const cameraMessage = document.getElementById("camera-message");
const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d");
const MAX_FRAME_SIDE = 1280;
let cameraStream = null;
let frameTimer = null;
let stateTimer = null;
let frameBusy = false;
let detections = [];

const categoryLabel = (category) => ({
  PLASTIC_RECYCLABLE: "Plastic - Recyclable",
  PLASTIC_NON_RECYCLABLE: "Plastic - Non-recyclable",
  PLASTIC_MANUAL_REVIEW: "Plastic - Manual review",
  METAL: "Metal",
  ORGANIC: "Organic"
}[category] || category);

function confidencePercent(value) {
  return value == null ? "N/A" : `${Math.round(Number(value) * 100)}%`;
}

function overlayLabel(item) {
  if (item.material !== "Plastic") {
    return `${item.material} | ${confidencePercent(item.detection_confidence)}`;
  }
  if (item.classification_state === "classified") {
    return `${categoryLabel(item.category)} | ${confidencePercent(item.detection_confidence)}`;
  }
  if (item.classification_state === "pending") {
    return `Plastic | Checking recyclability | ${confidencePercent(item.detection_confidence)}`;
  }
  if (item.classification_state === "uncertain" || item.classification_state === "failed") {
    return `Plastic | Manual review | ${confidencePercent(item.detection_confidence)}`;
  }
  return `Plastic | Awaiting confirmation | ${confidencePercent(item.detection_confidence)}`;
}

function resultLabel(item) {
  if (item.material === "Plastic") {
    if (item.classification_state === "pending") return "Plastic - Checking recyclability";
    if (item.classification_state === "uncertain" || item.classification_state === "failed") {
      return "Plastic - Manual review";
    }
    if (item.classification_state !== "classified") return "Plastic - Awaiting confirmation";
  }
  return categoryLabel(item.category);
}

function updateCameraMessage(items) {
  if (!items.length) {
    cameraMessage.textContent = "No supported waste object detected. Show a Plastic, Metal, or Organic item clearly.";
    return;
  }
  const plastic = items.find((item) => item.material === "Plastic");
  if (plastic && plastic.classification_state === "pending") {
    cameraMessage.textContent = "Plastic detected - checking recyclability.";
  } else if (plastic && ["uncertain", "failed"].includes(plastic.classification_state)) {
    cameraMessage.textContent = "Plastic detected, but recyclability is uncertain. Manual review required.";
  } else if (plastic && plastic.classification_state === "not_counted") {
    cameraMessage.textContent = "Object detected. Stabilizing the detection before classification.";
  } else {
    cameraMessage.textContent = `${items.length} supported waste object${items.length === 1 ? "" : "s"} detected.`;
  }
}

function renderState(state) {
  document.getElementById("count-total").textContent = state.total;
  document.getElementById("count-plastic").textContent = state.plastic_total;
  document.getElementById("count-recyclable").textContent = state.plastic_recyclable;
  document.getElementById("count-non-recyclable").textContent = state.plastic_non_recyclable;
  document.getElementById("count-review").textContent = state.plastic_manual_review;
  document.getElementById("count-metal").textContent = state.metal_total;
  document.getElementById("count-organic").textContent = state.organic_total;
  const rows = state.detections || state.items || [];
  if (!rows.length) {
    document.getElementById("detections").innerHTML = '<tr><td colspan="6">No detections yet.</td></tr>';
    return;
  }
  document.getElementById("detections").innerHTML = rows.map((item) =>
    `<tr><td>${escapeHtml(item.material)}</td><td>${escapeHtml(resultLabel(item))}</td>` +
    `<td>${confidencePercent(item.detection_confidence)}</td>` +
    `<td>${item.material !== "Plastic" ? "N/A" : confidencePercent(item.classification_confidence)}</td>` +
    `<td>${item.classification_inference_seconds == null ? "N/A" : Number(item.classification_inference_seconds).toFixed(1) + " s"}</td>` +
    `<td>${escapeHtml(item.reason)}</td></tr>`
  ).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

function drawOverlay() {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / frameCanvas.width;
  const scaleY = canvas.height / frameCanvas.height;
  for (const item of detections) {
    const [rawX1, rawY1, rawX2, rawY2] = item.box;
    const x1 = rawX1 * scaleX;
    const y1 = rawY1 * scaleY;
    const x2 = rawX2 * scaleX;
    const y2 = rawY2 * scaleY;
    const color = item.material === "Plastic" ? "#d97706" : item.material === "Metal" ? "#2563eb" : "#92400e";
    context.strokeStyle = color;
    context.lineWidth = Math.max(2, canvas.width / 500);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = overlayLabel(item);
    context.font = `${Math.max(14, canvas.width / 75)}px Segoe UI`;
    const y = Math.max(20, y1 - 6);
    context.fillStyle = color;
    context.fillRect(x1, y - 19, context.measureText(label).width + 8, 22);
    context.fillStyle = "white";
    context.fillText(label, x1 + 4, y - 2);
  }
}

async function pollFrame() {
  if (frameBusy || !cameraStream || video.readyState < 2) return;
  frameBusy = true;
  try {
    const scale = Math.min(
      1,
      MAX_FRAME_SIDE / Math.max(video.videoWidth, video.videoHeight)
    );
    frameCanvas.width = Math.round(video.videoWidth * scale);
    frameCanvas.height = Math.round(video.videoHeight * scale);
    frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.88));
    if (!blob) return;
    const response = await fetch("/api/frame", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream"},
      body: blob
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Frame analysis failed.");
    detections = data.detections;
    updateCameraMessage(detections);
    renderState(data);
    drawOverlay();
  } catch (error) {
    statusText.textContent = `Camera analysis: ${error.message}`;
  } finally {
    frameBusy = false;
  }
}

async function refreshState() {
  try {
    const response = await fetch("/api/state");
    const state = await response.json();
    renderState(state);
    if (state.camera_active && cameraStream) {
      detections = state.detections || [];
      updateCameraMessage(detections);
      drawOverlay();
      statusText.textContent = "Camera running";
    }
  } catch (error) {
    statusText.textContent = `Dashboard update failed: ${error.message}`;
  }
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  statusText.textContent = "Requesting camera access...";
  cameraMessage.textContent = "Waiting for camera permission...";
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Webcam access is unavailable. Open this app on localhost or in a secure browser context.");
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {facingMode: "environment"}, audio: false
    });
    video.srcObject = cameraStream;
    await video.play();
    statusText.textContent = "Loading detection model...";
    cameraMessage.textContent = "Camera connected. Loading the detection model...";
    const response = await fetch("/api/camera/start", {method: "POST"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not start detection.");
    stopButton.disabled = false;
    statusText.textContent = "Camera running";
    cameraMessage.textContent = "Camera ready. Looking for Plastic, Metal, or Organic waste...";
    frameTimer = setInterval(pollFrame, 1250);
    stateTimer = setInterval(refreshState, 1500);
    pollFrame();
  } catch (error) {
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    video.srcObject = null;
    statusText.textContent = `Could not start camera: ${error.message}`;
    cameraMessage.textContent = `Camera unavailable: ${error.message}`;
    startButton.disabled = false;
  }
});

stopButton.addEventListener("click", async () => {
  clearInterval(frameTimer);
  clearInterval(stateTimer);
  frameTimer = null;
  stateTimer = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;
  detections = [];
  context.clearRect(0, 0, canvas.width, canvas.height);
  cameraMessage.textContent = "Camera stopped. Click Start Camera to begin live detection.";
  try {
    const response = await fetch("/api/camera/stop", {method: "POST"});
    if (!response.ok) throw new Error("Server could not stop camera analysis.");
    renderState(await response.json());
    statusText.textContent = "Camera stopped and released";
  } catch (error) {
    statusText.textContent = error.message;
  }
  startButton.disabled = false;
  stopButton.disabled = true;
});

resetButton.addEventListener("click", async () => {
  const response = await fetch("/api/counts/reset", {method: "POST"});
  const state = await response.json();
  if (!response.ok) {
    statusText.textContent = state.error || "Could not reset counts.";
    return;
  }
  renderState(state);
  statusText.textContent = "Session counts reset";
});

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("analyze-image");
  button.disabled = true;
  statusText.textContent = "Analyzing uploaded image...";
  try {
    const response = await fetch("/api/upload", {method: "POST", body: new FormData(event.currentTarget)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Image analysis failed.");
    const result = document.getElementById("upload-result");
    result.src = data.annotated_image;
    result.style.display = "block";
    detections = data.detections;
    renderState(data);
    statusText.textContent = "Image analyzed; Plastic classification may finish asynchronously.";
    if (!stateTimer) stateTimer = setInterval(refreshState, 1500);
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

refreshState();
