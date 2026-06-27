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
model = None
model_load_error = None
model_lock = threading.Lock()


def load_model():
    global model, model_load_error
    try:
        from ultralytics import YOLO
        # yolov8n = "nano" = ~6MB, ~lowest RAM footprint of the YOLOv8 family.
        # yolov8m (the original choice) needs 700MB-1GB+ RAM to load, which
        # crashes Render's free tier (512MB RAM limit) with an OOM kill.
        # For a car-damage-specific model, swap this path with:
        #   YOLO("path/to/your/best.pt")
        m = YOLO("yolov8n.pt")  # auto-downloads on first run (~6MB)
        with model_lock:
            model = m
        logger.info("✅ YOLO model loaded successfully (yolov8n)")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"❌ Model load failed: {e}")


# Kick off model loading in the background immediately on import,
# so gunicorn can bind to the port right away.
threading.Thread(target=load_model, daemon=True).start()

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
        if model_load_error:
            return {"error": f"Model failed to load: {model_load_error}"}, 503
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
    return jsonify({
        "status": "ok",
        "service": "YOLO Vehicle Damage Detection API",
        "model": "YOLOv8n (COCO pretrained)",
        "model_loaded": model is not None,
        "version": "2.1.0"
    })


@app.route("/health", methods=["GET"])
def health():
    # Always return 200 fast so Render's health check / port-bind check
    # passes even while the model is still loading in the background.
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_load_error": model_load_error
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
