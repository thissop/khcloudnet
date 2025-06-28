import os
import cv2
import rasterio
from rasterio.transform import from_origin

def convert_png_to_tif(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                png_path = os.path.join(root, file)
                tif_path = png_path.replace('.png', '.tif')

                print(f"Converting {png_path} to {tif_path}")

                # Read PNG (including alpha if exists)
                img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

                # Check if image is grayscale or multi-channel
                if len(img.shape) == 2:  # Grayscale
                    channels = 1
                    img = img[:, :, None]  # Add channel dimension
                else:
                    channels = img.shape[2]

                height, width = img.shape[:2]

                with rasterio.open(
                    tif_path, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=channels,
                    dtype=img.dtype,
                    transform=from_origin(0, 0, 1, 1)  # Dummy geotransform
                ) as dst:
                    for i in range(channels):
                        dst.write(img[:, :, i], i + 1)

if __name__ == "__main__":
    base_dir = "/discover/nobackup/tkiker/data/khcloudnet"

    # Convert training images
    #convert_png_to_tif(os.path.join(base_dir, "train_images_10"))

    # Convert training masks
    #convert_png_to_tif(os.path.join(base_dir, "train_masks_10"))

    # Convert test images
    #convert_png_to_tif(os.path.join(base_dir, "test_images_10"))

    # Convert test masks
    convert_png_to_tif(os.path.join(base_dir, "test_masks_10"))

    print("All conversions completed.")
