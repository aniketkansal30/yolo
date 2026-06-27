from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import logging
import threading
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model Loading ────────────────────────────────────────────────────────────
# Loaded in a background thread so the web server can bind to $PORT
# immediately (Render's health check needs a fast response, otherwise it
# kills the instance before the model finishes loading).
#
# IMPORTANT: this must be resilient to gunicorn worker restarts. If the
# worker process that's loading the model gets killed/recycled before the
# thread finishes, a fresh module import happens and model_loaded can look
# stuck at False forever. ensure_model_loading() below is also called
# defensively from analyze_image(), so a stuck/failed load self-heals on
# the next request instead of staying broken forever.
model = None
model_load_error = None
model_loading = False
model_lock = threading.Lock()


def load_model():
    global model, model_load_error, model_loading
    try:
        from ultralytics import YOLO
        # yolov8n = "nano" = ~6MB, ~lowest RAM footprint of the YOLOv8 family.
        # yolov8m (the original choice) needs 700MB-1GB+ RAM to load, which
        # crashes Render's free tier (512MB RAM limit) with an OOM kill.
        # For a car-damage-specific model, swap this path with:
        #   YOLO("path/to/your/best.pt")
        m = YOLO("yolov8n.pt")  # auto-downloads on first run (~6MB)
        # Warm up with a tiny dummy inference so the very first real
        # request isn't the one that pays the JIT/graph-build cost.
        try:
            import numpy as np
            m(np.zeros((64, 64, 3), dtype="uint8"), verbose=False)
        except Exception as warmup_err:
            logger.warning(f"Warmup inference skipped: {warmup_err}")
        with model_lock:
            model = m
            model_load_error = None
        logger.info("✅ YOLO model loaded successfully (yolov8n)")
    except Exception as e:
        with model_lock:
            model_load_error = str(e)
        logger.error(f"❌ Model load failed: {e}")
    finally:
        with model_lock:
            model_loading = False


def ensure_model_loading():
    """Kick off a load attempt if nothing is loaded and nothing is in flight.
    Safe to call from any request handler — cheap no-op if already loaded
    or already loading."""
    global model_loading
    with model_lock:
        if model is not None or model_loading:
            return
        model_loading = True
    threading.Thread(target=load_model, daemon=True).start()


# Kick off model loading in the background immediately on import,
# so gunicorn can bind to the port right away.
ensure_model_loading()

# Car/vehicle related COCO class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Parts we map detections to (simulated from bounding box regions)
DAMAGE_PARTS = ["bumper", "hood", "door", "fender",
                "windshield", "headlight", "taillight", "trunk"]


def map_to_car_parts(boxes, img_width, img_height):
    """Map detected bounding boxes to car part names by position."""
    parts = set()
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_width    # normalized center x
        cy = (y1 + y2) / 2 / img_height   # normalized center y

        # Top half of image → hood / windshield / roof
        if cy < 0.4:
            parts.add("hood" if cx < 0.5 else "windshield")
        # Bottom region → bumper
        elif cy > 0.7:
            parts.add("bumper")
        # Left side
        elif cx < 0.3:
            parts.add("headlight" if cy < 0.55 else "door")
        # Right side
        elif cx > 0.7:
            parts.add("taillight" if cy < 0.55 else "fender")
        else:
            parts.add("door")

    return list(parts) if parts else ["bumper"]  # default if no boxes


def severity_from_confidence(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    elif conf >= 0.55:
        return "medium"
    return "low"


def analyze_image(image_base64: str):
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_w, img_h = img.size

    with model_lock:
        current_model = model

    if current_model is None:
        # Self-heal: kick off another load attempt in case the previous
        # one died silently (e.g. worker recycled mid-load). This won't
        # help THIS request, but the next request will likely succeed.
        ensure_model_loading()
        if model_load_error:
            return {"error": f"Model failed to load: {model_load_error}. Retrying in background — try again shortly."}, 503
        return {"error": "Model is still loading, please retry in a few seconds"}, 503

    results = current_model(img, verbose=False)[0]

    vehicle_boxes = []
    max_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in VEHICLE_CLASSES and conf > 0.3:
            coords = box.xyxy[0].tolist()
            vehicle_boxes.append(coords)
            max_conf = max(max_conf, conf)

    damage_detected = len(vehicle_boxes) > 0
    damaged_parts = []
    severity = "none"
    confidence = 0.0

    if damage_detected:
        damaged_parts = map_to_car_parts(vehicle_boxes, img_w, img_h)
        confidence = round(max_conf, 2)
        severity = severity_from_confidence(max_conf)

    return {
        "damage_detected": damage_detected,
        "damaged_parts": damaged_parts,
        "severity": severity,
        "confidence": confidence
    }, 200


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    with model_lock:
        loaded = model is not None
    return jsonify({
        "status": "ok",
        "service": "YOLO Vehicle Damage Detection API",
        "model": "YOLOv8n (COCO pretrained)",
        "model_loaded": loaded,
        "version": "2.1.1"
    })


@app.route("/health", methods=["GET"])
def health():
    # Always return 200 fast so Render's health check / port-bind check
    # passes even while the model is still loading in the background.
    # Also self-heals: if nothing is loaded and nothing is loading
    # (e.g. the original load thread died silently), kick off a retry.
    ensure_model_loading()
    with model_lock:
        loaded = model is not None
        loading = model_loading
        err = model_load_error
    return jsonify({
        "status": "healthy",
        "model_loaded": loaded,
        "model_loading": loading,
        "model_load_error": err
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "Missing field: image_base64"}), 400

        try:
            img_bytes = base64.b64decode(data["image_base64"])
            if len(img_bytes) < 100:
                return jsonify({"error": "Image too small or invalid"}), 400
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        result, status = analyze_image(data["image_base64"])
        return jsonify(result), status

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Local dev entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
