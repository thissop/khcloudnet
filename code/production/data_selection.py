import pandas as pd
from usgs_utils import download_browse_image
from tqdm import tqdm 
import os 

def download_browse_images_for_sampling(key_csv: str, n: int, output_csv:str, output_dir: str):
    if not os.path.exists(output_csv):
        key_df = pd.read_csv(key_csv)

        sampled_ids = set()  
        sampled_rows = []

        key_df = key_df.sample(frac=1, random_state=2026).reset_index(drop=True)

        while len(sampled_rows) < n:
            sample = key_df.sample(n=1)
            entity_ID = sample.iloc[0]['Entity ID']
            
            # Determine the conjugate camera suffix
            if entity_ID[-4] == 'A':
                conjugate_camera = 'F'
            elif entity_ID[-4] == 'F':
                conjugate_camera = 'A'
            else:
                raise Exception("Unexpected ID Format")
            
            # Create the conjugate ID
            conjugate_ID = entity_ID[: -4] + conjugate_camera + entity_ID[-3:]

            # Check if either this ID or its conjugate has already been added
            if entity_ID not in sampled_ids and conjugate_ID not in sampled_ids:
                # Extract the coordinates for distance checking
                nw_lat = sample.iloc[0]["NW Corner Lat dec"]
                sw_lat = sample.iloc[0]["SW Corner Lat dec"]
                nw_long = sample.iloc[0]["NW Corner Long dec"]
                ne_long = sample.iloc[0]["NE Corner Long dec"]

                # Check distance constraints
                if abs(abs(nw_lat) - abs(sw_lat)) <= 10 and abs(abs(nw_long) - abs(ne_long)) <= 10:
                    # Append sample if within the distance constraints
                    sampled_rows.append(sample)
                    sampled_ids.add(entity_ID)
                    sampled_ids.add(conjugate_ID)

            # key_df = key_df.drop(sample.index) --> This makes the job 5 times slower so I'm just going to not remove. 

        sampled_rows_df = pd.concat(sampled_rows, ignore_index=True)
        sampled_rows_df.to_csv(output_csv, index=False)

    else: 
        sampled_rows_df = pd.read_csv(output_csv)

    cols = ['Entity ID', 'Mission', 'Operations Number', 'Camera']
    sub_df = sampled_rows_df[cols]

    entity_IDs = list(sub_df[cols[0]])
    missions = list(sub_df[cols[1]])
    operations_numbers = list(sub_df[cols[2]])
    cameras = list(sub_df[cols[3]])

    download_browse_image(image_output_dir=output_dir, entity_IDs=entity_IDs, missions=missions, operations_numbers=operations_numbers, cameras=cameras)

def split_browse_images(input_dir:str, output_dir:str, scale_factor:int=1, cutout_size:int=None): 
    r'''
    
    if cutout_size is None, the image will be scaled by scale_factor, then cutout size becomes scaled image's height, so cutouts become [height, height] in dimensions
     
    '''
    
    import os 
    from PIL import Image
    import cv2

    Image.MAX_IMAGE_PIXELS = None
    
    for img in tqdm(os.listdir(input_dir)): 
        img_path = os.path.join(input_dir, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        
        if scale_factor != 1: 
            if scale_factor <= 0: 
                raise Exception('Scale Factor must be > 0')
            
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)  # Fast but lower quality
                        
        if cutout_size is None or cutout_size <= 0: 
            cutout_size = image.shape[0]    

        width = image.shape[1]
        height = image.shape[0]

        # Iterate over the data array to create cutouts
        row_index = 0
        for i in range(0, height, cutout_size):
            column_index = 0

            for j in range(0, width, cutout_size):
                cutout = image[i:i + cutout_size, j:j + cutout_size]

                # Check if the cutout meets the required dimensions
                if cutout.shape[0] < cutout_size or cutout.shape[1] < cutout_size:
                    # Skip saving if cutout is smaller than cutout_size x cutout_size
                    continue

                # Convert cutout to an image and save if it meets size requirements
                cutout_image = Image.fromarray(cutout)
                file_name = f'{img.split(".")[0]}-{row_index}_{column_index}.jpg'
                cutout_image.save(os.path.join(output_dir, file_name), compress_level=0)

                column_index += 1 

            row_index += 1

def plot_strips_on_map(input_csv: str, output_path: str): 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import Polygon
    from cartopy.io.img_tiles import Stamen
    
    # Load the CSV data
    df = pd.read_csv(input_csv)

    # Step 1: Define Polygons
    def create_polygon(row):
        try:
            return Polygon([
                (row["NW Corner Long dec"], row["NW Corner Lat dec"]),
                (row["NE Corner Long dec"], row["NE Corner Lat dec"]),
                (row["SE Corner Long dec"], row["SE Corner Lat dec"]),
                (row["SW Corner Long dec"], row["SW Corner Lat dec"]),
                (row["NW Corner Long dec"], row["NW Corner Lat dec"]) # Close the polygon
            ])
        except Exception as e:
            print(f"Error creating polygon for row {row.name}: {e}")
            return None  

    df["Polygon"] = df.apply(create_polygon, axis=1)
    df = df.dropna(subset=["Polygon"])  

    # Define extent based on data limits
    xmin, xmax = df["NW Corner Long dec"].min() - 5, df["NE Corner Long dec"].max() + 5
    ymin, ymax = df["SW Corner Lat dec"].min() - 5, df["NW Corner Lat dec"].max() + 5

    # Create Plot
    plt.figure(figsize=(12, 6))  
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot each polygon
    for poly in df["Polygon"]:
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='red', facecolor='red', alpha=0.7)

    ax.scatter(df["Center Longitude dec"], df["Center Latitude dec"], color='blue', marker='*', s=50)

    plt.title("Sampled Data Strips")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path, dpi=200)

def plot_summary_histograms(input_csv:str, strips_dir:str, output_path:str):
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import smplotlib 
    import os 
    import cv2 

    df = pd.read_csv(input_csv)

    dates = list(df['Acquisition Date'])
    
    months = [int(i.split('/')[1]) for i in dates]
    years = [int(i.split('/')[0]) for i in dates]

    lengths = []

    for img in os.listdir(strips_dir):
        img_path = os.path.join(strips_dir, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        lengths.append(image.shape[1])

    effective_long_spans = []
    for nw_long, ne_long, center_lat in zip(df['NW Corner Long dec'], df['NE Corner Long dec'], df['Center Latitude dec']):
        effective_long_span = np.abs(np.diff([float(nw_long), float(ne_long)]))*np.cos(np.radians(float(center_lat)))
        effective_long_spans.append(effective_long_span)

    # Make Plot

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(months)
    axs[0, 0].set(xlabel='Month', ylabel='Count')

    axs[0, 1].hist(years)
    axs[0, 1].set(xlabel='Year')
    axs[0, 1].ticklabel_format(style='plain', axis='x')

    axs[1, 0].hist(lengths)
    axs[1, 0].set(xlabel='Span (Pixels)', ylabel='Count')

    axs[1, 1].hist(np.array(effective_long_spans).flatten())
    axs[1, 1].set(xlabel='Span (Effective Longitude'+r"$^\circ$"+')')

    fig.tight_layout()
    plt.savefig(output_path)

def pick_training_cutouts(input_dir:str, output_dir:str, n_cutouts:int=4, zip_batch_dir:str=None, batch_size:int=80):
    r'''
    
    Note: this code assumes standard file structure in input_dir/output_dir of "EntityID-m_n.IMGTYPE", where EntityID is structured like "D3C1201-100004F017", "m" is the row number of the cutout, "n" is the column number of the cutout, and "IMGTYPE" is something like "png" or "jpeg", e.g.: "D3C1201-100004F017-0_1.png" 
    
    '''
    
    import os 
    import shutil 
    import random 
    import zipfile
    from tqdm import tqdm 
    import numpy as np

    if batch_size%2 != 0: 
        raise Exception("Batch Size Needs to be Divisible by 2")

    if n_cutouts % 2 != 0: 
        raise Exception("n_cutouts Must be Divisible by 2")

    entity_IDs = []

    for img in os.listdir(input_dir): 
        if img[0] != ".": # was getting annoying .DS_Store 
            img_list = img.split('-') # D3C1201-100045F008
            entity_ID = img_list[0]+'-'+img_list[1]
            entity_IDs.append(entity_ID)

    entity_IDs = list(set(entity_IDs))
    entity_IDs = random.sample(entity_IDs, len(entity_IDs))

    print('Selecting and Copying Cutouts for Annotation')

    selected_image_paths = []
    for entity_ID in tqdm(entity_IDs): 
        local_list = []
        for img in os.listdir(input_dir):
            if entity_ID in img: 
                local_list.append(img)
        
        list_to_copy = []
        len_local_list = len(local_list)
        if len_local_list == n_cutouts: 
            list_to_copy = local_list
        elif len_local_list > n_cutouts: 
            list_to_copy = random.sample(local_list, n_cutouts)

        for img in list_to_copy: 
            old_path = os.path.join(input_dir, img)
            new_path = os.path.join(output_dir, img)
            selected_image_paths.append(new_path)

            if not os.path.exists(new_path):
                shutil.copy(old_path, new_path)

    if zip_batch_dir is not None:
        print('Batching and Compressing Selected Cutouts')

        for i in tqdm(range(0, len(selected_image_paths), batch_size)):
            batch_files = selected_image_paths[i:i + batch_size]

            zip_path = os.path.join(zip_batch_dir, f'batch_{i // batch_size + 1}.zip')
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in batch_files:
                    file_path = os.path.join(output_dir, file)
                    zipf.write(file_path, arcname=os.path.basename(file))
    
### EVALUATE EVERYTHING ### 

#print('downloading browse images')
#key_csv = '/Users/tkiker/Documents/GitHub/khcloudnet/data/filtered_key_df.csv'
#n = 10000
#strips_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/strips'
#output_csv = '/Users/tkiker/Documents/GitHub/khcloudnet/data/sampled_rows.csv'
#download_browse_images_for_sampling(key_csv, n, output_csv, strips_dir)

#input_dir = strips_dir
output_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/cutouts'

#print('generating cutouts from browse images')
#split_browse_images(input_dir, output_dir)

#print('plotting strips on map')
#strips_map_path = '/Users/tkiker/Documents/GitHub/khcloudnet/code/personal/thaddaeus/monthly/oct2024/sampled-data-strip-overlaps.png'
#plot_strips_on_map(output_csv, strips_map_path)

#print('plotting histograms')
#histograms_plot_path = '/Users/tkiker/Documents/GitHub/khcloudnet/code/personal/thaddaeus/monthly/oct2024/sampled-data-histograms.png'
#plot_summary_histograms(output_csv, strips_dir, histograms_plot_path)

print('batching cutouts')
annotation_cutouts_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotation-cutouts'
zip_batch_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches'
pick_training_cutouts(input_dir=output_dir, output_dir=annotation_cutouts_dir, n_cutouts=4, zip_batch_dir=zip_batch_dir, batch_size=80)