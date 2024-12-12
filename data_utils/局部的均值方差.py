import os
from PIL import Image
import numpy as np

def calculate_statistics(image_folder, roi_folder):
    all_means = []
    all_stds = []

    # Traverse the image folder
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                # Construct the image and ROI file paths
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, image_folder)
                roi_path = os.path.join(roi_folder, relative_path)

                # Ensure the ROI file exists
                if os.path.exists(roi_path):
                    # Load the image and ROI
                    image = Image.open(image_path).convert('RGB')
                    roi = Image.open(roi_path).convert('L')  # Convert to grayscale for binary mask

                    image_np = np.array(image) / 255.0
                    roi_np = np.array(roi) /255.0

                    # Apply the mask
                    mask = roi_np > 0  # Assuming binary mask where 0 is background
                    masked_image = image_np[mask]

                    # Calculate mean and std
                    mean_rgb = np.mean(masked_image, axis=0)
                    std_rgb = np.std(masked_image, axis=0)

                    all_means.append(mean_rgb)
                    all_stds.append(std_rgb)

    # Calculate overall mean and std
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.std(all_stds, axis=0)

    return overall_mean, overall_std

# Usage
image_folder = '../datasets/con_datasets/changhai_suda2_485/diease_folder'
roi_folder = '../datasets/con_datasets/changhai_suda2_485/roi'
mean, std = calculate_statistics(image_folder, roi_folder)
print("Mean RGB:", mean)
print("Std RGB:", std)
