import streamlit as st
import numpy as np
import cv2
import os
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
    # Convert PIL image to OpenCV BGR
    img = np.array(pil_img.convert('RGB'))[:, :, ::-1].copy()

    # Preprocess
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Forward pass
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L_orig = cv2.split(lab)[0]
    # Reconstruct
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Ensure same height/width
    if colorized.shape[0] != img.shape[0] or colorized.shape[1] != img.shape[1]:
        colorized = cv2.resize(colorized, (img.shape[1], img.shape[0]))

    # Combine original (convert back to BGR->RGB for PIL)
    orig_rgb = img[:, :, ::-1]
    colorized_rgb = colorized[:, :, ::-1]

    # Add label area on top
    label_height = 40
    combined = np.hstack((orig_rgb, colorized_rgb))
    combined_with_labels = np.ones((combined.shape[0] + label_height, combined.shape[1], 3), dtype=np.uint8) * 255

    # Paste combined below label area
    combined_with_labels[label_height:, :] = combined

    # Put labels using OpenCV
    cv2.putText(combined_with_labels, "Original", (int(orig_rgb.shape[1]/4)-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(combined_with_labels, "Colorized", (orig_rgb.shape[1] + int(orig_rgb.shape[1]/4)-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Convert to PIL Image for display/download
    combined_pil = Image.fromarray(combined_with_labels)
    return combined_pil, Image.fromarray(orig_rgb), Image.fromarray(colorized_rgb)


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

    # If none, generate placeholders
    if not imgs:
        for i in range(4):
            w, h = thumb_size
            placeholder = Image.new('RGB', thumb_size, (230, 230, 250))
            d = ImageDraw.Draw(placeholder)
            txt = f"Dataset {i+1}"
            try:
                # Use system font where available
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
    st.set_page_config(page_title="Image Colorizer", layout='wide', page_icon='🎨')

    st.markdown("# 🎨 Image Colorization")
    st.markdown("Upload a grayscale or colored image and the model will colorize it. Compare side-by-side and download the result.")

    # Left column: dataset gallery and instructions
    left, right = st.columns([1,2])

    with left:
        st.subheader("Training dataset")
        # Prefer 'sampledat' folder if available, otherwise fall back to 'inputs'
        dataset_folder = 'sampledat' if os.path.exists('sampledat') else 'inputs'
        imgs = get_dataset_images(dataset_folder)

        # Button to view samples in a popup/modal (images only, no names)
        view_samples = st.button("View sample Train dataset")
        if view_samples:
            # Use modal when available (Streamlit >= 1.18). Fallback to expander otherwise.
            if hasattr(st, "modal"):
                with st.modal("Sample Training Dataset"):
                    st.markdown("### Sample images")
                    # Load images from the same dataset folder (show images only, no names)
                    sample_imgs = get_dataset_images(dataset_folder, max_images=52)
                    mcols = st.columns(3)
                    for i, (_, im) in enumerate(sample_imgs):
                        mcols[i % 3].image(im, use_container_width=True, caption=None)
            else:
                with st.expander("Sample Training Dataset"):
                    st.markdown("### Sample images")
                    sample_imgs = get_dataset_images(dataset_folder, max_images=52)
                    mcols = st.columns(3)
                    for i, (_, im) in enumerate(sample_imgs):
                        mcols[i % 3].image(im, use_container_width=True, caption=None)

    with right:
        st.subheader("Upload image")
        uploaded = st.file_uploader("Choose an image", type=['png','jpg','jpeg','bmp'])

        # Fallback to example in repo if present
        example_path = "vipul.png"
        use_example = False
        if uploaded is None and os.path.exists(example_path):
            use_example = st.checkbox("Use example image from repository (vipul.png)", value=False)

        if uploaded is None and not use_example:
            st.info("Upload an image to get started, or enable the example image.")

        # Load model lazily
        try:
            net = load_model()
        except Exception as e:
            st.error(f"Model files not found or failed to load: {e}")
            st.stop()

        if uploaded is not None or use_example:
            try:
                if uploaded is not None:
                    pil_img = Image.open(uploaded).convert('RGB')
                else:
                    pil_img = Image.open(example_path).convert('RGB')

                # Show original preview
                st.image(pil_img, caption='Original preview', use_container_width=True)

                # Colorize
                with st.spinner('Colorizing image...'):
                    combined_pil, orig_pil, colorized_pil = colorize_pil_image(pil_img, net)

                st.success('Colorization complete')
                st.subheader('Result')

                st.image(combined_pil, use_container_width=True)

                # Provide download button
                buf = pil_image_to_bytes(combined_pil, fmt='PNG')
                st.download_button('Download comparison image', data=buf, file_name='colorized_comparison.png', mime='image/png')

                # Also offer separate colorized image download
                buf2 = pil_image_to_bytes(colorized_pil, fmt='PNG')
                st.download_button('Download colorized image (no original)', data=buf2, file_name='colorized.png', mime='image/png')

            except Exception as e:
                st.error(f"Failed to process image: {e}")


if __name__ == '__main__':
    main()
