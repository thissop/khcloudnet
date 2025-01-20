def incorporate_thresholded_masks(threshold_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotated-batches/threshold', batch_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/annotated-batches', threshold_key_file:str='/Users/tkiker/Documents/GitHub/khcloudnet/code/monthly/jan2025/prepping-training-data/Thresholded Images Key.csv', background_image_batch_dir:str="/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/zipped-batches", output_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready', data_catalogue_csv:str='/Users/tkiker/Documents/GitHub/khcloudnet/data/sampled_rows.csv', image_strip_dir:str='/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/strips', plot_dir:str='/Users/tkiker/Documents/GitHub/khcloudnet/code/monthly/jan2025/prepping-training-data/plots', files_per_batch:int=80, output_mosaic_shape:list=[1536,1536], num_cutouts:int=9, cutouts_per_strip:int=4):
    '''
    
    move thresholded masks into corresponding directories; perform quality checks to make sure there are no duplicate/missing masks; prep everything for export to training; make summary statistics/plots
    
    Note: only works with png images (except strips which are jpg by default...fix this!)

    Note: Assumes that the process that creates cutouts and batches them was correct. 

    Future Improvement: get rid of zip in the whole process and just use tar

    fix so less args, standardize directory in which it's 

    '''
    
    import tarfile
    from geopy.geocoders import Nominatim
    import os
    import pandas as pd 
    from tqdm import tqdm 
    import shutil 
    import numpy as np
    from PIL import Image
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    import zipfile
    import cv2
    import scipy
    import smplotlib 

    # Extract Tar Files
    
    def extract_tar_files(): 
        for tar_file in tqdm(os.listdir(batch_dir)): 
            if 'khbatch' in tar_file and '.tar' in tar_file:
                extracted_path = os.path.join(batch_dir, tar_file.split('.')[0])
                
                if not os.path.exists(extracted_path):
                    with tarfile.open(os.path.join(batch_dir, tar_file), 'r') as tar:  # Use 'r:' for .tar, 'r:gz' for .tar.gz, 'r:bz2' for .tar.bz2
                        tar.extractall(path=extracted_path)  # Extract all contents to the specified directory

    extract_tar_files() 

    def clean_supervisely_directories(): 

        for khbatch_dir in os.listdir(batch_dir): 
            if "khbatch" in khbatch_dir and ".tar" not in khbatch_dir: 
                khbatch_dir = os.path.join(batch_dir, khbatch_dir)
                for obj in os.listdir(khbatch_dir): 
                    if 'dataset' not in obj and '.' not in obj: 
                        os.remove(os.path.join(khbatch_dir, obj))

                    elif 'dataset' in obj: 
                        dataset_dir = os.path.join(khbatch_dir, obj)

                        for sub_dir in os.listdir(dataset_dir):
                            if sub_dir!='masks_machine': 
                                shutil.rmtree(os.path.join(dataset_dir, sub_dir))

                            else: 
                                annotations_dir = os.path.join(dataset_dir, sub_dir)
                                
                                for annotation_file in os.listdir(annotations_dir): 
                                    old_img_path = os.path.join(annotations_dir, annotation_file)
                                    new_img_path = os.path.join(khbatch_dir, annotation_file)
                                    if os.path.exists(new_img_path): 
                                        os.remove(old_img_path)

                                    else: 
                                        shutil.copyfile(old_img_path, new_img_path)

                        shutil.rmtree(dataset_dir)

                    if 'png' not in obj: 
                        os.remove(os.path.join(khbatch_dir, obj))
        
    clean_supervisely_directories()
    
    # Quality Control for Missing/Duplicate Images

    key_df=pd.read_csv(threshold_key_file)
    
    threshold_batch_arr = key_df['Batch'].to_list()
    threshold_status_arr = key_df['Status'].to_list()
    threshold_name_arr = key_df['Name'].to_list()
    threshold_overlay_arr = key_df['Overlay'].to_list()

    # Clean Threshold Dir of Non-Relevant Threshold Images
    for file in os.listdir(threshold_dir): 
        if '.png' in file: 
            found = False 
            img_name = file.split('-cloud')[0]
            for name in threshold_name_arr: 
                if img_name == name: 
                    found = True
                    break
            
            if not found:
                os.remove(os.path.join(threshold_dir, file)) 

    # Q.C. I: Check for Duplicate Entries in Key 
    unique_elements, counts = np.unique(threshold_name_arr, return_counts=True)
    duplicates = unique_elements[counts > 1]

    if len(duplicates)!=0: 
        print(duplicates)
        raise Exception('Exception: Duplicate Entries in Key')

    # Q.C. II: Check for Key Items Missing Files

    threshold_file_names = [i.split('-cloud')[0] for i in os.listdir(threshold_dir)]
    missing = np.setxor1d(threshold_file_names, unique_elements)
    missing = missing[missing!='.DS_Store']
    
    if len(missing)!=0: 
        print(missing)
        raise Exception('Exception: mis-match between key and files in threshold.')

    # Copy Threshold Images into 
    
    for name, batch, status, overlay in zip(threshold_name_arr, threshold_batch_arr, threshold_status_arr, threshold_overlay_arr): 
        threshold_file_path = os.path.join(threshold_dir, f'{name}-cloud-mask.png')

        specific_batch_dir = os.path.join(batch_dir, f'khbatch{batch}')

        batch_file_path = os.path.join(specific_batch_dir, f'{name}.png')

        # Merge Supervisely and Threshold Results
        if os.path.exists(batch_file_path) and os.path.exists(threshold_file_path) and overlay: 
            threshold_image_arr = np.array(Image.open(threshold_file_path).convert('L'))
            batch_image_arr = np.array(Image.open(batch_file_path).convert('L'))

            mask = batch_image_arr<0.5
            batch_image_arr[mask] = 0
            batch_image_arr[~mask] = 255

            mask = threshold_image_arr > 100
            threshold_image_arr[mask] = 255
            threshold_image_arr[~mask] = 0

            merged = np.where(np.logical_or(batch_image_arr == 255, threshold_image_arr == 255), 255, 0)

            #os.remove(threshold_file_path)
            os.remove(batch_file_path)

            # Save the image as PNG without compression
            Image.fromarray(merged.astype(np.uint8)).save(threshold_file_path, format="PNG", compress_level=0)

        shutil.copyfile(threshold_file_path, batch_file_path)

    # Q.C. III: Check that there are no missing images
    for khbatch_dir in tqdm(os.listdir(batch_dir)): 
        if "khbatch" in khbatch_dir and ".tar" not in khbatch_dir: 
            khbatch_dir = os.path.join(batch_dir, khbatch_dir)
            number_of_files = len([i for i in os.listdir(khbatch_dir) if '.png' in i])
            if number_of_files!=files_per_batch:
                #print(number_of_files)
                #print(khbatch_dir)

                batch_number = khbatch_dir.split('batch')[-1]

                img_dir = os.path.join(background_image_batch_dir, f'batch_{batch_number}')
                if not os.path.exists(img_dir): 
                    with zipfile.ZipFile(f'{img_dir}.zip', 'r') as zip_ref:
                        zip_ref.extractall(path=img_dir)

                all_image_names = [i for i in os.listdir(img_dir) if '.png' in i]
                
                names_of_images_with_annotations = [i for i in os.listdir(khbatch_dir) if '.png' in i]

                missing = np.setxor1d(all_image_names, names_of_images_with_annotations)

                batch_name_ = khbatch_dir.split("/")[-1]

                raise Exception(f'Batch {batch_number} is Missing File(s): {missing}')

            # Make all Annotation Masks [0,255] 
            
            def make_in_correct_color_range():
                for mask_image in os.listdir(khbatch_dir):
                    if 'png' in mask_image:
                        mask_image_path = os.path.join(khbatch_dir, mask_image)
                        img_arr = np.array(Image.open(mask_image_path).convert('L'))

                        mask = img_arr<0.5
                        img_arr[mask] = 0
                        img_arr[~mask] = 255

                        Image.fromarray(img_arr.astype(np.uint8)).save(mask_image_path, format="PNG", compress_level=0)

            make_in_correct_color_range()

    # Calculate Statistics & Distribute the Masks/Images
    
    """
    - Distribution of cloud sizes (and total) in pixel and km^2 terms (approximate)

    - Plot of where the images are from (centroids)

    - % of images that don't have annotation (add this to spread sheet to return)

    """

    num_rows = int(num_cutouts**0.5)
    if num_cutouts%num_rows != 0: 
        raise Exception('sqrt(num_cutouts) needs to be an integer for square cutouts!')

    image_names = []
    cloud_pixel_counts = [] # --> turn this into ground coverage as well. 
    areas = []
    ground_coverages = []

    mask_cutout_dir = os.path.join(output_dir, 'masks')
    background_cutout_dir = os.path.join(output_dir, 'images')

    original_cutout_length = 1

    for khbatch_dir in tqdm(os.listdir(batch_dir)): 
        if "khbatch" in khbatch_dir and ".tar" not in khbatch_dir: 
            batch_number = khbatch_dir.split('batch')[-1]

            img_dir = os.path.join(background_image_batch_dir, f'batch_{batch_number}') # standardize this to tar! 
            if not os.path.exists(img_dir): 
                with zipfile.ZipFile(f'{img_dir}.zip', 'r') as zip_ref:
                    zip_ref.extractall(path=img_dir)

            khbatch_dir = os.path.join(batch_dir, khbatch_dir)
            for mask_image_name in os.listdir(khbatch_dir):
                if "png" in mask_image_name:
                    mask_image_path = os.path.join(khbatch_dir, mask_image_name)
                    background_image_path = os.path.join(background_image_batch_dir, f'batch_{batch_number}/{mask_image_name}')

                    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
                    background_image = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
                    image_names.append(mask_image_name.split('-')[0]+'-'+mask_image_name.split('-')[1])

                    original_cutout_length = mask_image.shape[0]

                    # Do Calculations
                    # Do Connected Components to calculate how many pixels 

                    non_zero_sum = np.count_nonzero(mask_image)
                    cloud_pixel_counts.append(non_zero_sum)

                    labeled_array, num_features = scipy.ndimage.label(mask_image)

                    unique_labels, counts = np.unique(labeled_array, return_counts=True)

                    if len(counts) > 1: # Ignore the background label (0)
                        areas.extend(counts[1:].tolist())  # Remove background label

                    #_, _, stats, _ = cv2.connectedComponentsWithStats(mask_image)
                    #areas.extend(stats[1:, cv2.CC_STAT_AREA].tolist()) # re-introduce ability to blacken out very small connected components. 

                    # Resize, Sample, and Save

                    # modify this so it only runs when the images don't exist 
                    try: 
                        resized_mask_image = cv2.resize(mask_image, output_mosaic_shape, interpolation=cv2.INTER_NEAREST)
                        resized_background_image = cv2.resize(background_image, output_mosaic_shape, interpolation=cv2.INTER_AREA)
                    except: 
                        print(mask_image_name)
                        quit()

                    # Generate Cutouts --> This needs to get moved to a standard raster_utils.py production routine. 

                    cutout_size = int(output_mosaic_shape[0]/num_rows)

                    file_prefix = mask_image_name.split('.')[0]
                        
                    row_index = 0
                    for i in range(0, output_mosaic_shape[0], cutout_size): # this might/should be modified for non uniform shapes (rectangles, only works for squares for now)
                        column_index = 0

                        for j in range(0, output_mosaic_shape[0], cutout_size):
                            
                            file_name = f'{file_prefix}-{row_index}_{column_index}.png' 
                            mask_cutout_image_path = os.path.join(mask_cutout_dir, file_name)
                            background_cutout_image_path = os.path.join(background_cutout_dir, file_name)

                            if not os.path.exists(mask_cutout_image_path) and not os.path.exists(background_cutout_image_path): 

                                mask_cutout = resized_mask_image[i:i + cutout_size, j:j + cutout_size]
                                background_cutout = resized_background_image[i:i + cutout_size, j:j + cutout_size]

                                mask_cutout_image = Image.fromarray(mask_cutout)
                                background_cutout_image = Image.fromarray(background_cutout)

                                file_prefix = mask_image_name.split('.')[0]
                                file_name = f'{file_prefix}-{row_index}_{column_index}.png' 

                                mask_cutout_image.save(mask_cutout_image_path, compress_level=0)
                                background_cutout_image.save(background_cutout_image_path, compress_level=0)

                            column_index += 1 

                        row_index += 1 


    # check that same amount of files in both directories. 

    # Estimate Ground Coverage and Make Summary Plots

    plt.hist(areas)
    plt.xlabel('Cutout Annotation Area (Pixels)')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'coverage-hist-pixelcount.png'), dpi=300)
    plt.clf()

    coverage_percentages = 100*np.array(cloud_pixel_counts)/original_cutout_length**2
    plt.hist(coverage_percentages)
    plt.xlabel(r'Cutout Cloud Cover (%)')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'coverage-hist-percent.png'), dpi=300)
    plt.clf()

    catalogue_df = pd.read_csv(data_catalogue_csv)

    covered_area_kmsqs = []
    covered_area_percentages = []
    center_lats = []
    center_longs = []

    #if len(list(set(image_names)))!=

    image_names = list(set(image_names))

    for image_name in list(set(image_names)):
        catalogue_row = catalogue_df.loc[catalogue_df['Entity ID'] == image_name].squeeze()
        total_area_kmsq = catalogue_row['area_kmsq_equal_area']
        center_lats.append(catalogue_row['Center Latitude dec'])
        center_longs.append(catalogue_row['Center Longitude dec'])

        cutout_image_path = os.path.join(image_strip_dir, f'{image_name}.jpg')

        cutout_image_path = f'/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/strips/{image_name}.jpg'
        cutout_image = Image.open(cutout_image_path)
        area_factor = cutouts_per_strip*cutout_image.height/cutout_image.width

        covered_area_kmsqs.append(area_factor*total_area_kmsq)
        covered_area_percentages.append(100*area_factor)

    plt.scatter(covered_area_kmsqs, covered_area_percentages)
    plt.xlabel(r'Sampled Area per Strip (km$^2$)')
    plt.xlim(0, 0.2*25000)
    plt.ylabel(r'Percent of Strip Sampled')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'coverage-scatter.png'), dpi=300)
    plt.clf()
        
    plt.figure(figsize=(12, 4))

    # Create map with PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Extent of the map 
    xmin, xmax, ymin, ymax = -180, 180, 0, 90 
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

    ax.scatter(center_longs, center_lats, color='red', transform=ccrs.PlateCarree(), s=10)
    # Geographical features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    #plt.title("Locations of Image Strips")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'globalmap-scatter.png'), dpi=300)
    plt.clf()

    # INTERESTING TIDBITS 

    # Mean/Std of Coverage 

    print(f'Mean (%): {np.mean(coverage_percentages)}, STD (%): {np.std(coverage_percentages)}')

    # % with less than 5% cloud Coverage 

    print(f'% with Less than 5% Cloud Coverage: {100*len(np.where(coverage_percentages<5)[0])/len(coverage_percentages)}')

    # Total Coverage 
    print(f'{np.sum(covered_area_kmsqs)}')

    # most covered one: 

    print('One with the most coverage km^2: ', np.max(covered_area_kmsqs), image_names[np.argmax(covered_area_kmsqs)])

incorporate_thresholded_masks()

'''
D3C1206-100081F063-0_0.png is in batch 80 and 66? 
'''