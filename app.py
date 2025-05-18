import streamlit as st  # Streamlit for building the web app interface
import tensorflow as tf  # TensorFlow for deep learning operations
import tensorflow_hub as hub  # TensorFlow Hub for loading pre-trained models
import numpy as np  # NumPy for numerical operations
from PIL import Image  # PIL for image processing
import io  # IO for handling byte streams (used in download)

# Load the pre-trained style transfer model from TensorFlow Hub
# This is cached to avoid reloading on every run, improving performance
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Call the function to load the model once
model = load_model()

# Preprocess the image to make it compatible with the model
def preprocess(image_data, target_size=(256, 256)):
    img = Image.open(image_data).convert("RGB")  # Open and convert image to RGB
    img = img.resize(target_size)  # Resize image to expected model input size
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, height, width, 3)
    return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to TensorFlow tensor

# Title and description of the app
st.title("🎨 Neural Style Transfer App")
st.write("Upload a content image and a style image, and this app will apply artistic style transfer!")

# File upload widgets for content and style images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# Proceed only if both images are uploaded
if content_file and style_file:
    # Preprocess both images
    content_img = preprocess(content_file)
    style_img = preprocess(style_file)

    # Display the uploaded content image
    st.subheader("Content Image")
    st.image(content_file, use_column_width=True)

    # Display the uploaded style image
    st.subheader("Style Image")
    st.image(style_file, use_column_width=True)

    # When user clicks the button, apply the style
    if st.button("Apply Style"):
        with st.spinner("Stylizing..."):  # Show spinner during processing
            # Use the model to perform style transfer
            output = model(content_img, style_img)[0]  # Output is a tensor with shape (1, h, w, 3)
            output_image = tf.squeeze(output).numpy()  # Remove batch dimension
            output_image = (output_image * 255).astype(np.uint8)  # Convert from float [0,1] to uint8 [0,255]
            output_pil = Image.fromarray(output_image)  # Convert numpy array to PIL Image

            # Display the final stylized image
            st.subheader("Stylized Image")
            st.image(output_pil, use_column_width=True)

            # Prepare image for download
            buf = io.BytesIO()  # Create in-memory binary stream
            output_pil.save(buf, format="JPEG")  # Save image to stream in JPEG format
            # Add download button
            st.download_button("Download Stylized Image", data=buf.getvalue(),
                               file_name="stylized_output.jpg", mime="image/jpeg")
