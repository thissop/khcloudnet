import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO
import os

def autoannotate_clouds(input_image, cutoff_level: int, min_size:int=75):
    """
    Generate a binary mask for cloud regions based on a cutoff level.
    """

    data = np.array(input_image.convert('L'))  # Convert to grayscale

    # Create a binary mask based on the cutoff level
    binary_mask = np.where(data > cutoff_level, 255, 0).astype(np.uint8)
    
    # Apply morphological operations (closing) to smooth out the mask
    kernel = np.ones((5, 5), np.uint8) 
    binary_mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Detect connected components
    num_labels, labels_im = cv2.connectedComponents(binary_mask_closed)

    # Calculate the size of each connected component
    component_sizes = np.bincount(labels_im.flatten())

    filtered_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):  # Start from 1 to skip the background label
        if component_sizes[label] >= min_size:
            filtered_mask[labels_im == label] = 255  # Keep this component

    return filtered_mask 

def plot_intensity_histogram(data, cutoff_level):
    flattened_data = data.flatten()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(flattened_data, bins=50, color='black', edgecolor='k', fill=False)
    ax.hist(flattened_data, bins=20, alpha=0.6, color='gray')

    # Vertical line for cutoff level
    ax.axvline(x=cutoff_level, color='red', linestyle='--', linewidth=2, label=f'Cutoff Level: {cutoff_level}')

    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Count')
    ax.set_title('Intensity Histogram')

    ax.legend()
    
    st.pyplot(fig)

def overlay_images(original_image, binary_mask):
    # Convert to RGBA
    original_image = original_image.convert("RGBA")
    
    # Create a mask with white (255, 255, 255) for mask areas and transparent background
    mask_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)  # Initialize RGBA mask
    mask_rgba[binary_mask == 255, :3] = [255, 0, 0]  # Red for mask
    mask_rgba[binary_mask == 255, 3] = 75  # Fully opaque where mask exists
    mask_rgba[binary_mask == 0, 3] = 0  # Fully transparent for background

    mask_rgba_img = Image.fromarray(mask_rgba, 'RGBA')

    # Overlay the mask on top of the original image with transparency
    blended_image = Image.alpha_composite(original_image, mask_rgba_img)
    return blended_image

st.title("Cloud Auto-Annotate and Histogram App")

# Step 1: Upload image
uploaded_file = st.file_uploader("Choose an image (e.g., satellite/cloud image)...", type=["png", "jpg", "jpeg"])

# Step 2: Upload CSV for cloud-specific annotations
csv_file = st.file_uploader("Upload CSV file with cloud annotation points (x, y, class)...", type=["csv"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    input_file_name = os.path.splitext(uploaded_file.name)[0] 
    mask_file_name = f"{input_file_name}-cloud-mask.png"

    grayscale_image = np.array(image.convert('L'))

    st.subheader("Histogram of Intensity Values (Cloud Detection)")

    max_pixel_value = int(grayscale_image.max())
    slider_value = st.slider("Cutoff Level for Clouds", int(max_pixel_value*0.1), max_pixel_value, int(0.6*max_pixel_value))

    # Update histogram dynamically as the slider moves
    plot_intensity_histogram(grayscale_image, slider_value)

    # Add a button to recalculate the mask
    if st.button("Calculate Cloud Mask"):
        # Step 3: Generate the binary mask for clouds
        binary_mask = autoannotate_clouds(image, slider_value)

        # If CSV is uploaded, remove points connected to irrelevant regions
        if csv_file is not None:
            csv_data = pd.read_csv(csv_file)
            binary_mask = remove_river_regions(binary_mask, csv_data)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            mask_img = Image.fromarray(binary_mask)
            st.image(mask_img, caption="Binary Mask (Clouds)", use_column_width=True)

        # Step 4: Overlay the mask on the original image and display it below
        st.subheader("Overlay of Original Image and Cloud Mask")
        overlay_image = overlay_images(image, binary_mask)
        st.image(overlay_image, caption="Original Image with Cloud Mask Overlay", use_column_width=True)

        st.subheader("Download Cloud Binary Mask")
        mask_img = Image.fromarray(binary_mask)

        buf = BytesIO()
        mask_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Cloud Binary Mask",
            data=byte_im,
            file_name=mask_file_name,
            mime="image/png"
        )
