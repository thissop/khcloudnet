def get_connected_components(mask_path):
    r"""Compute connected components in binary mask and return pixel area sizes for each component."""
    from PIL import Image 
    import numpy as np
    import cv2 

    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    binary_mask = (mask_np == 255).astype(np.uint8)

    # Apply connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Extract area sizes
    area_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background 

    return area_sizes

def plot_image_mask(image_path, mask_path, plot_dir:str='code/monthly/feb2025/tweaking-data/plots'):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import smplotlib 
    import os 

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  

    image_np = np.array(image)
    mask_np = np.array(mask)

    # RGBA overlay
    overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255  # Red channel
    overlay[..., 3] = (mask_np / 255) * 128  # Alpha channel (0.5 of 255 is 128)

    # Plot image with overlay

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(image_np)
    axs[0].imshow(overlay, cmap='Reds', alpha=overlay[..., 3] / 255.0)
    axs[0].axis("off")

    axs[1].hist(get_connected_components(mask_path))
    axs[1].set(xscale='log', xlabel='Pixel Area', ylabel='Count')

    fig.tight_layout()
    image_name = image_path.split('/')[-1]
    plt.savefig(os.path.join(plot_dir, image_name))

import numpy as np 
import os 
from tqdm import tqdm 

image_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/test_images'

for filename in tqdm(np.random.choice(os.listdir(image_dir), size=100, replace=False)):
    image_path = f'{image_dir}/{filename}'
    mask_path = f'/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/test_masks/{filename}'
    plot_image_mask(image_path, mask_path)