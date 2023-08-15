# %%
import torch
from PIL import Image
from mmseg.apis import MMSegInferencer
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_image(img_path, target_size=(1024, 1024)):
    """Load and resize an image to the given target size."""
    original_img = Image.open(img_path)
    original_shape = original_img.size
    resized_img = original_img.resize(target_size)
    return np.array(original_img), np.array(resized_img), original_shape

# Load the original and resized images
img_path = '/workspace/demo.jpg'
original_img_array, img, original_shape = load_and_resize_image(img_path)

# Inference
inferencer = MMSegInferencer(model='ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024')
out = inferencer(img, show=True)
predictions = out['predictions'] # [1024, 1024]

# Reshape predictions back to the original shape
predictions = predictions.astype(np.uint8)
predictions_resized = Image.fromarray(predictions).resize(original_shape, Image.NEAREST)
predictions_resized = np.array(predictions_resized)

# Plot the original image and the prediction_resized side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img_array)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predictions")
plt.imshow(predictions_resized, cmap='jet')  # using a colormap for visualization of predictions
plt.axis('off')

plt.tight_layout()
plt.show()


# %%
