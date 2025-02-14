import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import smplotlib 
import os 

image_path = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/cutouts/D3C1215-300947A006-0_2.png'
image = Image.open(image_path).convert("L")
image = np.array(image).astype(np.uint8)  # Ensure uint8 format


# Apply CLAHE (Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
image_clahe = clahe.apply(image)

image_pil = Image.fromarray(image_clahe)

enhancer = ImageEnhance.Contrast(image_pil)
image_enhanced = enhancer.enhance(2.5)  

image_enhanced = np.array(image_enhanced)

output_dir = '/Users/tkiker/Desktop'

cv2.imwrite(os.path.join(output_dir, image_path.split('/')[-1]), image_enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])

image = np.flipud(np.fliplr(image))
image = image[image.shape[0]//2:, 0:]
image_enhanced = np.flipud(np.fliplr(image_enhanced))
image_enhanced = image_enhanced[image.shape[0]//2:, 0:]


fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original")

axs[1].imshow(image_enhanced, cmap='gray')
axs[1].set_title("Enhanced")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_frame_on(False)

plt.tight_layout()

#plt.savefig('code/monthly/feb2025/contrast-demonstration.png', dpi=300)