import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import io

# Paths to model files (assumed to be in the same folder as this app)
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"


@st.cache_resource
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


def get_dataset_images(folder='inputs', max_images=6, thumb_size=(240,160)):
    imgs = []
    if os.path.exists(folder):
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    p = os.path.join(folder, fname)
                    im = Image.open(p).convert('RGB')
                    im.thumbnail(thumb_size)
                    imgs.append((fname, im))
                    if len(imgs) >= max_images:
                        break
                except Exception:
                    continue

    # Placeholder images if none found
    if not imgs:
        for i in range(4):
            w, h = thumb_size
            placeholder = Image.new('RGB', thumb_size, (230, 230, 250))
            d = ImageDraw.Draw(placeholder)
            txt = f"Dataset {i+1}"
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = None
            tw, th = d.textsize(txt, font=font)
            d.text(((w-tw)/2,(h-th)/2), txt, fill=(40,40,40), font=font)
            imgs.append((f"placeholder_{i+1}.png", placeholder))
    return imgs


def pil_image_to_bytes(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title="Image Colorizer (Hybrid Comparison)", layout='wide', page_icon='🎨')
    st.title("🎨 Image Colorization — Normal vs Hybrid Comparison")
    st.markdown("Upload a grayscale image to compare **Normal** vs **hybrid ** colorization results.")

    uploaded = st.file_uploader("Choose an image", type=['png','jpg','jpeg','bmp'])
    example_path = "vipul.png"
    use_example = False
    if uploaded is None and os.path.exists(example_path):
        use_example = st.checkbox("Use example image from repository (vipul.png)", value=False)

    if uploaded is None and not use_example:
        st.info("Upload an image or enable example image to start.")
        st.stop()

    try:
        net = load_model()
    except Exception as e:
        st.error(f"Model files not found or failed to load: {e}")
        st.stop()

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert('RGB')
    else:
        pil_img = Image.open(example_path).convert('RGB')

    st.subheader("Original Image")
    st.image(pil_img, use_container_width=True)

    with st.spinner("Running colorization pipelines..."):
        # First method
        single_pass_output = colorize_pil_image(pil_img, net)
        # Second method (hybrid)
        first_pass, hybrid_output = colorize_pil_image_hybrid(pil_img, net)

    st.success("✅ Colorization complete")

    # Display results side-by-side
    st.subheader("Comparison Results")
    c1, c2, c3 = st.columns(3)
    c1.image(pil_img, caption="Original", use_container_width=True)
    c2.image(single_pass_output, caption="Colorization", use_container_width=True)
    c3.image(hybrid_output, caption="Hybrid Colorization", use_container_width=True)

    # Downloads
    st.download_button(
        "⬇️ Download Single-Pass Output",
        data=pil_image_to_bytes(single_pass_output),
        file_name="colorized_single.png",
        mime="image/png",
    )

    st.download_button(
        "⬇️ Download Hybrid Output",
        data=pil_image_to_bytes(hybrid_output),
        file_name="colorized_hybrid.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
