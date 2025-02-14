import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm 
import smplotlib 

# Directories
image_dir = "/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/images"
mask_dir = "/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/masks"

# LBP Parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

def extract_features(image):
    """Extract LBP + Gradient (Sobel) features from grayscale image."""
    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method="uniform")

    # Gradient Magnitude using Sobel filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return lbp, gradient_magnitude

def segment_clouds_kmeans(image):
    """Segment clouds using K-Means (k=3) on brightness + texture + gradient."""
    lbp, gradient = extract_features(image)
    features = np.column_stack((image.flatten(), lbp.flatten(), gradient.flatten()))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Assume the cluster with the highest average brightness is clouds
    cluster_means = [np.mean(image.flatten()[labels == i]) for i in range(3)]
    cloud_cluster = np.argmax(cluster_means)
    
    return (labels.reshape(image.shape) == cloud_cluster).astype(np.uint8) * 255

def compute_cloud_coverage(mask):
    """Compute cloud coverage percentage based on mask (true or predicted)."""
    cloud_pixels = np.sum(mask > 128)  # Count white pixels in the mask
    total_pixels = mask.size
    return (cloud_pixels / total_pixels) * 100

# Lists to store results
true_coverage_list = []
predicted_kmeans_list = []

# Process image-mask pairs
for filename in tqdm(os.listdir(image_dir)[0:100]):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            # Load grayscale image and mask
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Compute true cloud coverage from mask
            true_coverage = compute_cloud_coverage(mask)
            true_coverage_list.append(true_coverage)

            # Compute predicted cloud coverage using K-Means
            segmented_kmeans = segment_clouds_kmeans(image)
            predicted_kmeans = compute_cloud_coverage(segmented_kmeans)
            predicted_kmeans_list.append(predicted_kmeans)


# Plot True vs. Predicted Cloud Coverage (K-Means & GMM)
plt.figure(figsize=(10, 5))

# K-Means Plot

plt.scatter(true_coverage_list, predicted_kmeans_list, alpha=0.7, edgecolors='k', label="K-Means")
plt.plot([0, 100], [0, 100], 'r--', label="Perfect Prediction")
plt.xlabel("True Cloud Coverage (%)")
plt.ylabel("Predicted (K-Means) Cloud Coverage (%)")
plt.title("True vs. Predicted Cloud Coverage (K-Means)")
plt.legend()
plt.grid(True)
plt.show()