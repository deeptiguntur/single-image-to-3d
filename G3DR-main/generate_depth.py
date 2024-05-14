import os
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the MiDaS model
model_type = "DPT_Large"     # MiDaS v3 - Large (highest accuracy, slowest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms based on model type
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# Path to read images from
image_path = "/home/deept/single-image-to-3d/G3DR-main/images_train/images"

# Output folder to save processed images
output_folder = "/home/deept/single-image-to-3d/G3DR-main/dataset3/elephant_256_with_depth"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all image files in the specified path
for filename in os.listdir(image_path):
    if filename.endswith((".jpg")):
        file_path = os.path.join(image_path, filename)
        
        # Read the image
        img = cv2.imread(file_path)
        input_filename = os.path.splitext(filename)[0] + ".jpg"
        input_path = os.path.join(output_folder, input_filename)
        cv2.imwrite(input_path, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform the image
        input_batch = transform(img).to(device)
        
        # Perform inference
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert prediction to numpy array
        output = prediction.cpu().numpy()

        output_scaled = (output / np.max(output)) * np.iinfo(np.uint16).max
        output_scaled = output_scaled.astype(np.uint16)

        
        # Save the output with the desired filename format in the output folder
        output_filename = os.path.splitext(filename)[0] + "_depth.png"

        output_path = os.path.join(output_folder, output_filename)
        # # cv2.imwrite(output_path, output)
        # # plt.imsave(output_path, output, vmin=0, vmax=1, cmap='binary', format='png', dpi=100, metadata=None, pil_kwargs={'bits': 16})
        # plt.imsave(output_path, output, cmap='binary')
        cv2.imwrite(output_path, output_scaled)
