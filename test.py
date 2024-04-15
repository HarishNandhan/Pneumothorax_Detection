import numpy as np
import requests
import cv2
from utils import encode_img, decode_img
import matplotlib.pyplot as plt
import json

def input_gen(image_name):
    """
    Makes data in proper input format.
    """
    image = cv2.imread(image_name)
    if image is None:
        raise ValueError(f"Failed to read image at {image_name}")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = encode_img(image)
    data = {
        "image": image
    }
    data = json.dumps(data)
    return data


def output_gen(resp):
    """
    Process output data.
    """
    data = resp.json()
    image = data['data']['image']
    image = decode_img(image)
    return image

img_path = "data/a4.png"
data = input_gen(img_path)
resp = requests.post("http://localhost:4001/predict", data)
out = output_gen(resp)

# Load the original image
original_image = cv2.imread(img_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Resize the original image to match the output
original_image_resized = cv2.resize(original_image, (out.shape[1], out.shape[0]))

# Convert the segmented image to a binary mask
binary_mask = (out > 0.5).astype(np.uint8)

# Overlay the segmented region on the original image
alpha = 0.5  # Opacity of the overlay
overlay = original_image_resized.copy()
overlay[binary_mask == 1] = [255, 0, 0]  # Highlight pneumothorax region in red

# Check if pneumothorax is detected
if np.sum(binary_mask) > 0:
    title = "Pneumothorax is Detected"
else:
    title = "No pneumothorax"

# Display the overlay image with title
plt.imshow(overlay)
plt.title(title)  # Add the title
plt.axis('off')
plt.show()
