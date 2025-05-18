import tensorflow as tf  # Core TensorFlow library
import tensorflow_hub as hub  # For loading pre-trained models from TF Hub
import numpy as np  # Numerical operations
from PIL import Image  # Image loading and saving
import matplotlib.pyplot as plt  # Displaying images with matplotlib

# Function to load and preprocess image
def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert("RGB")  # Open image and ensure it's RGB
    img = img.resize(target_size)  # Resize to match model input size (256x256)
    img = np.array(img) / 255.0  # Normalize pixel values from [0, 255] to [0.0, 1.0]
    img = np.expand_dims(img, axis=0)  # Add a batch dimension (shape becomes [1, height, width, channels])
    return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to a TensorFlow tensor
# Local paths to content and style images
content_path = "C:/Users/BhargaviMandala/Downloads/flower.jpg"
style_path = "C:/Users/BhargaviMandala/Downloads/flowerart.jpg"

# Load and preprocess the images
content_image = load_image(content_path)
style_image = load_image(style_path)

print("Loading Style Transfer Model...")
# Load pre-trained style transfer model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Apply style transfer: model returns a stylized image tensor
stylized_image = model(content_image, style_image)[0]  # Get the output tensor from the returned tuple
# Postprocess the output tensor to convert to displayable image
output_image = tf.squeeze(stylized_image).numpy()  # Remove batch dimension (shape: [height, width, channels])
output_image = (output_image * 255).astype(np.uint8)  # Convert from float32 [0.0, 1.0] to uint8 [0, 255]

# Convert numpy array to PIL Image for saving
output_pil = Image.fromarray(output_image)

# Define output path and save the final stylized image
output_path = "C:/Users/BhargaviMandala/Downloads/stylized_output.jpg"
output_pil.save(output_path)
print(f"Stylized image saved to: {output_path}")

# Display the final stylized image using matplotlib
plt.imshow(output_pil)
plt.title("Stylized Image")
plt.axis("off")  # Hide axis ticks and labels
plt.show()








