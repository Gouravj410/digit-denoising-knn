import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from joblib import load
from io import BytesIO

# Load trained model
model = load("knn_cleaner_model.joblib")

st.set_page_config(page_title="Digit Denoiser", layout="centered")
st.title("🧹 Digit Image Denoiser")

st.markdown("""
Upload a **noisy digit image** (28x28 grayscale) and let the model denoise it!
""")

uploaded_file = st.file_uploader("Upload your digit image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array_flat = img_array.reshape(1, -1)

    # Predict (denoise)
    cleaned = model.predict(img_array_flat)
    denoised_image = cleaned[0].reshape(28, 28)

    # Convert back to image for display and download
    denoised_img_pil = Image.fromarray((denoised_image * 255).astype(np.uint8))

    # Show side-by-side
    st.subheader("🖼️ Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_array, caption="Noisy Input", use_column_width=True)

    with col2:
        st.image(denoised_image, caption="Denoised Output", use_column_width=True)

    # Convert to bytes and create download button
    buf = BytesIO()
    denoised_img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Denoised Image",
        data=byte_im,
        file_name="denoised_digit.png",
        mime="image/png"
    )
