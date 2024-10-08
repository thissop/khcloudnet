# khcloudnet
 
## Project Summary

Between the 1960s and 1980s, the CIA captured thousands of extraordinarily high-resolution (approximately 2-4 feet per pixel) panchromatic images across the globe. This data, stored in the form of original film rolls, was recently declassified, but remains neither fully digitized nor georeferenced. Although the USGS, the government agency responsible for the database, has made low-resolution scans of these films freely available online, high-resolution scans are only available upon request at a cost of around $30 per film roll.

One major challenge researchers face when using this database is selecting which film rolls to purchase. The USGS does not provide estimates of cloud cover on the film strips, forcing researchers to manually inspect each image to ensure it is usable, a time-consuming and inefficient process.

Our project aims to address this issue by analyzing the low-resolution scans to automatically estimate cloud cover. By doing so, we can help researchers more easily identify high-quality film strips with minimal cloud coverage, making their selection process more efficient and cost-effective. We will achieve this by downloading a subset of the low-resolution "browse" scans and training a UNet machine learning model to segment and detect cloud coverage in the images. Once we have trained our UNet model, we will calculate the cloud coverage on the entireity of the dataset, publish our results, and contact USGS for them to incorporate our results into their data selection portal. 

## Project Plan 

1. Image Strip Selection:

I made you a catalog of metadata for all images taken in the Declass III database (total of ~500k, you can access this .csv file at `data/declassiii-catalog.csv`). The metadata includes the following columns:

Entity ID
Acquisition Date
Mission
Frame Number
Image Type
Camera
Camera Resolution
Corner coordinates in longitude/latitude for each image.

Your first task is to filter the data by removing all rows where the images were taken outside of the summer months (June-August) or have a Camera Resolution that is not "2 to 4 feet." Once you’ve applied these filters, you’ll focus on creating a sampling strategy to ensure we have a geographically and temporally stratified set of images for training. You’ll define the exact selection criteria based on your ideas, exploration, and our discussions.

2. Image Processing:

Once the relevant image strips are selected, you’ll use the following Python function to download the low-resolution "browse" images for those strips. The function takes the filtered metadata (specifically the Entity ID, Mission, Operation Number, and Camera columns) and downloads the images as JPEG files, which you’ll use to create the training data. You’ll store the downloaded images in the specified directory and ensure they are ready for the next phase of processing.

You will use the Python function `download_browse_image()` to download the browse images (the function is located in the file `code/production/usgs_utils.py`), which will form the basis for your cutouts and cloud segmentation work.

3. Cutout Creation and Cloud Labeling:

After downloading the relevant browse images, you’ll use the PIL library to resize the images and create the 1024 x 1024 pixel cutouts. Each strip is 1920 pixels tall with variable lengths, so we’ll standardize them by shrinking both dimensions by a factor of 1.875. Here's how you can achieve that using the PIL library:

```python
from PIL import Image

# Open the image
img = Image.open(image_path)

# Resize image (shrink by a factor of 1.875)
width, height = img.size
new_size = (int(width / 1.875), int(height / 1.875))
resized_img = img.resize(new_size)
```

This process will allow you to resize the images, which will then be split up into cutouts (think double for loop) and be labeled for cloud regions using the Supervisely platform. These labeled images will serve as training data for the UNet model, which I will train on a high-performance NASA computer.

4. Future Work:

Once we’ve built a reliable model for cloud detection, we’ll move on to a follow-up project. This second phase will involve optimizing the selection of film strips to minimize the number of strips a researcher needs to purchase to cover a specific area over a given time period. The success of this phase will depend on the accuracy of the cloud cover estimates produced in the first phase.

Tools and Setup:

- Python: relevant libraries include pandas, geopandas, numpy, matplotlib, PIL, etc. 
- VS Code: for code development and project management. 
- QGIS: for geospatial analysis and working with coordinate data.
- EarthExplorer (USGS): make sure to create an account at EarthExplorer to access image metadata and download additional resources.