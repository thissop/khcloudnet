import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

file_path = '/Users/tkiker/Downloads/coords - Sheet1.csv' 
data = pd.read_csv(file_path)

def create_polygon(row):
    corners = [
        (row['NW Corner Long dec'], row['NW Corner Lat dec']),
        (row['NE Corner Long dec'], row['NE Corner Lat dec']),
        (row['SE Corner Long dec'], row['SE Corner Lat dec']),
        (row['SW Corner Long dec'], row['SW Corner Lat dec']),
        (row['NW Corner Long dec'], row['NW Corner Lat dec']) 
    ]
    return Polygon(corners)

polygons = data.apply(create_polygon, axis=1)

gdf = gpd.GeoDataFrame(data, geometry=polygons)

gdf.set_crs(epsg=4326, inplace=True)

output_gpkg_path = 'D3C1203-100056A002_outline.gpkg'  # Define your output path
gdf.to_file(output_gpkg_path, driver='GPKG')

