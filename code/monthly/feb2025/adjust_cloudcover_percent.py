def edge_detection(image_path, output_dir): 
    import os 
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    def detect_black_edges(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image was loaded correctly
        if image is None:
            raise ValueError("Error: Unable to load image. Check file path.")
        
        # Convert image to binary using adaptive thresholding
        _, binary_adaptive = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours of the largest non-black region
        contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Handle case where no contours are found
        if not contours:
            plt.figure(figsize=(6, 6))
            plt.imshow(binary_adaptive, cmap='gray')
            plt.title("No contours found. Check thresholding.")
            plt.show()
            raise ValueError("No contours detected. Check image content or thresholding.")
        
        # Find bounding box of the largest contour (assuming it's the image content)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Refine left edge detection using a moving window approach
        col_sums = np.sum(binary_adaptive, axis=0)  # Sum along columns

        # Define a more strict threshold to avoid stopping on small bright areas (e.g., text)
        threshold = np.max(col_sums) * 0.3  # Adjust to be stricter
        left_edge = x  # Default to bounding box edge
        for i in range(x, len(col_sums)):  # Start searching from initial left bound x
            if col_sums[i] > threshold:  # Find first column where intensity sum exceeds threshold
                left_edge = i
                break

        # Keep right, top, and bottom edges as detected
        right_edge = x + w

        # Adjust vertical bounds based on the further one from the image border
        img_height = image.shape[0]
        top_distance = y  # Distance of detected top edge from image top
        bottom_distance = img_height - (y + h)  # Distance of detected bottom edge from image bottom
        max_distance = max(top_distance, bottom_distance)
        top_edge = max_distance
        bottom_edge = img_height - max_distance

        return left_edge, right_edge, top_edge, bottom_edge

    # Path to the uploaded image
    left_edge, right_edge, top_edge, bottom_edge = detect_black_edges(image_path)

    # Load image for visualization
    image_color = cv2.imread(image_path)
    cv2.line(image_color, (left_edge, 0), (left_edge, image_color.shape[0]), (0, 0, 255), 2)  # Left edge
    cv2.line(image_color, (right_edge, 0), (right_edge, image_color.shape[0]), (255, 0, 0), 2)  # Right edge
    cv2.line(image_color, (0, top_edge), (image_color.shape[1], top_edge), (0, 255, 0), 2)  # Top edge
    cv2.line(image_color, (0, bottom_edge), (image_color.shape[1], bottom_edge), (255, 255, 0), 2)  # Bottom edge

    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Edges - Left: {left_edge}, Right: {right_edge}, Top: {top_edge}, Bottom: {bottom_edge}")
    save_path = os.path.join(output_dir, image_path.split('/')[-1])
    plt.savefig(save_path)

    plt.clf()

from tqdm import tqdm 
image_paths = ['/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_80/D3C1212-300564A013-0_4.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_80/D3C1213-100145F003-0_0.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_80/D3C1213-100159A034-0_4.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_79/D3C1201-400415A042-0_0.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_77/D3C1201-100007F013-0_14.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_76/D3C1215-401134A002-0_0.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_75/D3C1217-200765F006-0_0.png',
               '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches/batch_74/D3C1206-100197A003-0_4.png',
               '']
output_dir = '/Users/tkiker/Documents/GitHub/khcloudnet/code/monthly/feb2025/edge-detection'

for image_path in tqdm(image_paths): 
    edge_detection(image_path, output_dir)