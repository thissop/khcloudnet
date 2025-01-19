# khcloudnet
 
## Project Summary

Between the 1960s and 1980s, the CIA captured thousands of extraordinarily high-resolution (approximately 2-4 feet per pixel) panchromatic images across the globe. This data, stored in the form of original film rolls, was recently declassified, but remains neither fully digitized nor georeferenced. Although the USGS, the government agency responsible for the database, has made low-resolution scans of these films freely available online, high-resolution scans are only available upon request at a cost of around $30 per film roll.

One major challenge researchers face when using this database is selecting which film rolls to purchase. The USGS does not provide estimates of cloud cover on the film strips, forcing researchers to manually inspect each image to ensure it is usable, a time-consuming and inefficient process.

Our project aims to address this issue by analyzing the low-resolution scans to automatically estimate cloud cover. By doing so, we can help researchers more easily identify high-quality film strips with minimal cloud coverage, making their selection process more efficient and cost-effective. We will achieve this by downloading a subset of the low-resolution "browse" scans and training a UNet machine learning model to segment and detect cloud coverage in the images. Once we have trained our UNet model, we will calculate the cloud coverage on the entireity of the dataset, publish our results, and contact USGS for them to incorporate our results into their data selection portal. 