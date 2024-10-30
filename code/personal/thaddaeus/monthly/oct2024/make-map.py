import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
from matplotlib.colors import LogNorm
from rtree import index  # For spatial indexing
import smplotlib 
from tqdm import tqdm 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('/Users/tkiker/Documents/GitHub/khcloudnet/data/filtered_key_df.csv')

# Step 1: Define Polygons
def create_polygon(row):
    try:
        return Polygon([
            (row["NW Corner Long dec"], row["NW Corner Lat dec"]),
            (row["NE Corner Long dec"], row["NE Corner Lat dec"]),
            (row["SE Corner Long dec"], row["SE Corner Lat dec"]),
            (row["SW Corner Long dec"], row["SW Corner Lat dec"]),
            (row["NW Corner Long dec"], row["NW Corner Lat dec"]) # Need to repeat first coordinate to close polygon
        ])
    except Exception as e:
        print(f"Error creating polygon for row {row.name}: {e}")
        return None  

df["Polygon"] = df.apply(create_polygon, axis=1)
df = df.dropna(subset=["Polygon"])  

# Step 2: Create Grid 
xmin, ymin, xmax, ymax = -180, 0, 180, 90  # Only over northern hemisphere for this first step 
n_cells = 500 # n_cells^2 = n_grid_boxes; w = 360/500=0.72∘, h = 90/500 = 0.18∘, so at equator, area ≈ 1.6km^2, whereas at 45deg lat, area ≈ 1.1km^2
x_bins = np.linspace(xmin, xmax, n_cells)
y_bins = np.linspace(ymin, ymax, n_cells)

overlap_counts = np.zeros((n_cells-1, n_cells-1), dtype=int)

# Step 3: Calculate Overlaps 

# Use R-tree spatial index to speed up intersection checks

'''

> R-tree is spatial indexing structure that organizes geometric data, such as polygons or bounding boxes, to make spatial queries (like intersections or nearest neighbor searches) more efficient. 

> The way it works is by inserting the bounding box of every grid cell into the tree, and then grouping these bounding boxes together recursively, forming a tree structure where each node represents a bounding box. 

> Without R-tree optimization, naive method would require n_cells^2 number of comparisons per browse iamge polygon 

> However, now that I think about it, we could do without R-tree or other spatial indexing methods if we just calculating slices of the grid cells matrix to do comparisons within (since grid cells matrix is evenly spaced). 

'''

spatial_index = index.Index()

# Insert each grid cell into the spatial index
cell_polygons = []
for i in tqdm(range(n_cells - 1)):
    for j in range(n_cells - 1):
        cell = box(x_bins[i], y_bins[j], x_bins[i+1], y_bins[j+1])
        spatial_index.insert(i * (n_cells - 1) + j, cell.bounds)
        cell_polygons.append(cell)

for poly in tqdm(df["Polygon"]):
    # Reduce grid cell search space by only looking through those that are in the polygon's bounding box
    possible_matches_index = list(spatial_index.intersection(poly.bounds))  
    possible_matches = [cell_polygons[idx] for idx in possible_matches_index]

    # Count actual overlaps
    for cell in possible_matches:
        if poly.intersects(cell):
            # Calculate grid index of the cell
            i = np.searchsorted(x_bins, cell.bounds[0]) - 1
            j = np.searchsorted(y_bins, cell.bounds[1]) - 1
            overlap_counts[j, i] += 1

# Step 4: Plot Overlap Heatmap over World Map

plt.figure(figsize=(12, 4))  
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

img = ax.imshow(overlap_counts, extent=[xmin, xmax, ymin, ymax], origin="lower", 
                cmap="hot", norm=LogNorm(), transform=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

cbar = plt.colorbar(img, ax=ax, label="Number of Overlaps", fraction=0.035, pad=0.02, shrink=0.6)

plt.title("Data Strip Overlaps")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig('code/personal/thaddaeus/monthly/oct2024/data-strip-overlaps.png')