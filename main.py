import numpy as np
import cv2
import os

# Paths to model files
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Input image path
input_path = "vipul.png"
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found. Check path.")

# Extract filename and extension
filename = os.path.basename(input_path)
name, ext = os.path.splitext(filename)

# Read and preprocess the image
image = cv2.imread(input_path)

if image is None:
    raise ValueError("Failed to read the image.")

if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif image.shape[2] == 4:
    image = image[:, :, :3]  # Drop alpha

scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Forward pass
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Combine images side by side
# Ensure same height
if colorized.shape[0] != image.shape[0]:
    colorized = cv2.resize(colorized, (image.shape[1], image.shape[0]))

combined = np.hstack((image, colorized))

# Add labels on top of each image
label_height = 40
combined_with_labels = np.ones((combined.shape[0] + label_height, combined.shape[1], 3), dtype=np.uint8) * 255

# Write text labels
cv2.putText(combined_with_labels, "Original", (int(image.shape[1]/4)-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(combined_with_labels, "Colorized", (image.shape[1] + int(image.shape[1]/4)-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

# Paste combined image below labels
combined_with_labels[label_height:, :] = combined

# Save final combined image
combined_path = os.path.join(output_dir, f"combined_{name}{ext}")
cv2.imwrite(combined_path, combined_with_labels)

print(f"✅ Combined image saved successfully:")
print(f"   • {combined_path}")
