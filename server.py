import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Paths to model files (assumed to be in the same folder as this app)
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"
EXAMPLE_IMAGE = "vipul.png"

app = Flask(__name__)
CORS(app)


def load_model():
    # Load network and cluster centers
    if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL) or not os.path.exists(POINTS):
        missing = [p for p in (PROTOTXT, MODEL, POINTS) if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing model files: {missing}")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # setup cluster centers as network blobs
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


def colorize_pil_image(pil_img, net):
    img = np.array(pil_img.convert('RGB'))[:, :, ::-1].copy()
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return Image.fromarray(colorized[:, :, ::-1])  # Convert BGR → RGB for PIL


def colorize_pil_image_hybrid(pil_img, net):
    """Hybrid method: run the colorization model twice."""
    # First pass
    first_pass = colorize_pil_image(pil_img, net)

    # Convert first pass output to grayscale and run second pass
    gray_second = first_pass.convert("L").convert("RGB")
    second_pass = colorize_pil_image(gray_second, net)

    return first_pass, second_pass


def pil_image_to_datauri(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/{fmt.lower()};base64,{b64}"


# Load model once at startup
try:
    net = load_model()
    print("Model loaded successfully")
except Exception as e:
    net = None
    print(f"Warning: failed to load model at startup: {e}")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": net is not None})


@app.route("/colorize", methods=["POST"])
def colorize():
    global net
    if net is None:
        try:
            net = load_model()
        except Exception as e:
            return jsonify({"error": f"Model not available: {e}"}), 500

    use_example = request.form.get('use_example', 'false').lower() == 'true'

    if use_example:
        if not os.path.exists(EXAMPLE_IMAGE):
            return jsonify({"error": f"Example image {EXAMPLE_IMAGE} not found on server."}), 400
        pil_img = Image.open(EXAMPLE_IMAGE).convert('RGB')
    else:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        f = request.files['image']
        pil_img = Image.open(f.stream).convert('RGB')

    # Run colorization
    single_pass_output = colorize_pil_image(pil_img, net)
    first_pass, hybrid_output = colorize_pil_image_hybrid(pil_img, net)

    return jsonify({
        "original": pil_image_to_datauri(pil_img, fmt='PNG'),
        "single": pil_image_to_datauri(single_pass_output, fmt='PNG'),
        "hybrid": pil_image_to_datauri(hybrid_output, fmt='PNG')
    })


if __name__ == "__main__":
    # Run on port 5000 by default
    app.run(host='0.0.0.0', port=5000, debug=False)
