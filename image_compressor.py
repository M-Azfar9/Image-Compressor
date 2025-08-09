import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import io

# Streamlit App
st.title("Image Compression with K-Means Clustering")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Select number of colors
K = st.slider("Select number of colors (K)", min_value=2, max_value=64, value=16)

if uploaded_file is not None:
    # Loading image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Converting image to numpy array
    img_array = np.array(image)

    # Reshaping to (num_pixels, 3) for RGB
    pixels = img_array.reshape(-1, 3)

    # Applying KMeans
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(pixels)
    new_colors = kmeans.cluster_centers_.astype(np.uint8)

    # Replacing each pixel with its cluster's color
    compressed_pixels = new_colors[labels]
    compressed_img = compressed_pixels.reshape(img_array.shape)

    # Displaying compressed image
    st.image(compressed_img, caption=f"Compressed Image ({K} colors)", use_column_width=True)

    # Option to download compressed image
    compressed_pil = Image.fromarray(compressed_img)
    buf = io.BytesIO()
    compressed_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Compressed Image",
        data=byte_im,
        file_name=f"compressed_{K}_colors.png",
        mime="image/png"
    )
