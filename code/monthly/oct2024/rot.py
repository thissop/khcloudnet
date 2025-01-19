import numpy as np
import rasterio
from rasterio.transform import Affine

# Define the input and output paths
input_path = '/Volumes/My Passport for Mac/khdata/usgs-geotiff/D3C1203-100056A002/D3C1203-100056A002_h.tif'
output_path = '/Volumes/My Passport for Mac/khdata/usgs-geotiff/D3C1203-100056A002/D3C1203-100056A002_h_rotated.tif'

# Open the original GeoTIFF file
with rasterio.open(input_path) as src:
    # Read the image data into a numpy array
    image_data = src.read()
    
    # Rotate the image by 180 degrees using numpy
    rotated_data = np.flip(np.flip(image_data, axis=1), axis=2)  # Flip vertically and horizontally

    # Get the original affine transform and update it for the rotated image
    original_transform = src.transform
    new_transform = Affine(
        original_transform.a, original_transform.b, 
        original_transform.c + (original_transform.a * src.width),
        original_transform.d, original_transform.e, 
        original_transform.f + (original_transform.e * src.height)
    )

    # Prepare profile for output with adjusted metadata
    profile = src.profile
    profile.update({
        "driver": "GTiff",
        "height": rotated_data.shape[1],
        "width": rotated_data.shape[2],
        "transform": new_transform,
        "crs": src.crs  # Ensure CRS is copied correctly
    })

    # Write the rotated data to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rotated_data)

print("Rotation and save completed.")
